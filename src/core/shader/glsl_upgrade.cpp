#include "core/shader/glsl_upgrade.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <vector>

namespace wmgl::shader {
namespace {

bool isIdentStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
bool isIdent(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

// `text` with the identifier `name` (whole words) replaced by `with`.
std::string replaceIdentifier(const std::string& text, const std::string& name, const std::string& with) {
    std::string out;
    size_t from = 0;
    for (size_t at = text.find(name); at != std::string::npos; at = text.find(name, at + name.size())) {
        if ((at > 0 && isIdent(text[at - 1])) || (at + name.size() < text.size() && isIdent(text[at + name.size()]))) continue;
        out.append(text, from, at - from);
        out += with;
        from = at + name.size();
    }
    out.append(text, from, std::string::npos);
    return out;
}

// Compatibility built-ins that become declared variables. The declarations
// use the same names with a wmgl_ prefix; the compatibility profile (W22)
// binds the inputs to the conventional vertex arrays and fills the uniform
// block from fixed-function state.
struct BuiltinDecl {
    const char* name;
    const char* decl;   // declaration of wmgl_<name>
    int stages;         // bitmask of Stage values where it applies
};
constexpr int kVS = 1 << static_cast<int>(Stage::Vertex);
constexpr int kFS = 1 << static_cast<int>(Stage::Fragment);
constexpr int kGS = 1 << static_cast<int>(Stage::Geometry);

const BuiltinDecl kVertexInputs[] = {
    {"gl_Vertex", "in vec4 wmgl_Vertex;", kVS},
    {"gl_Normal", "in vec3 wmgl_Normal;", kVS},
    {"gl_Color", "in vec4 wmgl_Color;", kVS},
    {"gl_SecondaryColor", "in vec4 wmgl_SecondaryColor;", kVS},
    {"gl_FogCoord", "in float wmgl_FogCoord;", kVS},
    {"gl_MultiTexCoord0", "in vec4 wmgl_MultiTexCoord0;", kVS},
    {"gl_MultiTexCoord1", "in vec4 wmgl_MultiTexCoord1;", kVS},
    {"gl_MultiTexCoord2", "in vec4 wmgl_MultiTexCoord2;", kVS},
    {"gl_MultiTexCoord3", "in vec4 wmgl_MultiTexCoord3;", kVS},
    {"gl_MultiTexCoord4", "in vec4 wmgl_MultiTexCoord4;", kVS},
    {"gl_MultiTexCoord5", "in vec4 wmgl_MultiTexCoord5;", kVS},
    {"gl_MultiTexCoord6", "in vec4 wmgl_MultiTexCoord6;", kVS},
    {"gl_MultiTexCoord7", "in vec4 wmgl_MultiTexCoord7;", kVS},
};

// Varyings between vertex and fragment stages.
const BuiltinDecl kVaryings[] = {
    {"gl_FrontColor", "out vec4 wmgl_FrontColor;", kVS},
    {"gl_BackColor", "out vec4 wmgl_BackColor;", kVS},
    {"gl_FrontSecondaryColor", "out vec4 wmgl_FrontSecondaryColor;", kVS},
    {"gl_BackSecondaryColor", "out vec4 wmgl_BackSecondaryColor;", kVS},
    {"gl_FogFragCoord", "out float wmgl_FogFragCoord;", kVS},
    {"gl_ClipVertex", "vec4 wmgl_ClipVertex;", kVS},
};

const char* kFixedFunctionUniforms[] = {
    "gl_ModelViewMatrix", "gl_ProjectionMatrix", "gl_ModelViewProjectionMatrix", "gl_TextureMatrix",
    "gl_NormalMatrix", "gl_ModelViewMatrixInverse", "gl_ProjectionMatrixInverse",
    "gl_ModelViewProjectionMatrixInverse", "gl_TextureMatrixInverse", "gl_ModelViewMatrixTranspose",
    "gl_ProjectionMatrixTranspose", "gl_ModelViewProjectionMatrixTranspose", "gl_TextureMatrixTranspose",
    "gl_ModelViewMatrixInverseTranspose", "gl_ProjectionMatrixInverseTranspose",
    "gl_ModelViewProjectionMatrixInverseTranspose", "gl_TextureMatrixInverseTranspose", "gl_NormalScale",
    "gl_ClipPlane", "gl_Point", "gl_FrontMaterial", "gl_BackMaterial", "gl_LightSource", "gl_LightModel",
    "gl_FrontLightModelProduct", "gl_BackLightModelProduct", "gl_FrontLightProduct", "gl_BackLightProduct",
    "gl_TextureEnvColor", "gl_EyePlaneS", "gl_EyePlaneT", "gl_EyePlaneR", "gl_EyePlaneQ", "gl_ObjectPlaneS",
    "gl_ObjectPlaneT", "gl_ObjectPlaneR", "gl_ObjectPlaneQ", "gl_Fog",
};

}  // namespace

GlslVersion parseGlslVersion(const std::string& source) {
    GlslVersion v;
    size_t i = 0;
    const size_t n = source.size();
    // Skip whitespace and comments before #version.
    while (i < n) {
        if (std::isspace(static_cast<unsigned char>(source[i]))) { ++i; continue; }
        if (source.compare(i, 2, "//") == 0) { while (i < n && source[i] != '\n') ++i; continue; }
        if (source.compare(i, 2, "/*") == 0) {
            const size_t end = source.find("*/", i + 2);
            i = end == std::string::npos ? n : end + 2;
            continue;
        }
        break;
    }
    if (i < n && source[i] == '#') {
        size_t j = i + 1;
        while (j < n && (source[j] == ' ' || source[j] == '\t')) ++j;
        if (source.compare(j, 7, "version") == 0) {
            v.declared = true;
            j += 7;
            const size_t lineEnd = source.find('\n', j);
            std::istringstream line(source.substr(j, lineEnd == std::string::npos ? std::string::npos : lineEnd - j));
            int number = 0;
            std::string profile;
            line >> number >> profile;
            if (number) v.number = number;
            v.es = profile == "es";
            v.compatibility = profile == "compatibility";
            v.core = profile == "core";
        }
    }
    if (v.number <= 130 && !v.es) v.compatibility = true;
    return v;
}

std::string esToDesktopGlsl(const std::string& preprocessed, int version) {
    if (version >= 310) {
        // glslang's lowering of atomic counters (relaxed Vulkan rules)
        // subtracts int constants from uints, which GLSL ES allows only with
        // EXT_shader_implicit_conversions. The shader has passed GLSL ES's
        // checks already: codegen may convert.
        const size_t at = preprocessed.find("#version");
        if (at == std::string::npos) return preprocessed;
        const size_t end = preprocessed.find('\n', at);
        if (end == std::string::npos) return preprocessed;
        return preprocessed.substr(0, end + 1) + "#extension GL_EXT_shader_implicit_conversions : enable\n" +
               preprocessed.substr(end + 1);
    }
    if (version != 100) return preprocessed;
    // GLSL ES 1.00 extensions' names for desktop GLSL 1.20 functions and
    // variables, and its built-in constants in vectors (desktop GLSL 1.20
    // counts components).
    static const std::map<std::string, std::string> kRename = {
        {"texture2DLodEXT", "texture2DLod"},         {"texture2DProjLodEXT", "texture2DProjLod"},
        {"textureCubeLodEXT", "textureCubeLod"},     {"texture2DGradEXT", "texture2DGradARB"},
        {"textureCubeGradEXT", "textureCubeGradARB"}, {"texture2DProjGradEXT", "textureProjGrad"},
        {"shadow2DEXT", "shadow2D"},                 {"shadow2DProjEXT", "shadow2DProj"},
        {"gl_FragDepthEXT", "gl_FragDepth"},         {"gl_MaxDrawBuffersEXT", "gl_MaxDrawBuffers"},
        {"gl_MaxVaryingVectors", "(gl_MaxVaryingFloats / 4)"},
        {"gl_MaxVertexUniformVectors", "(gl_MaxVertexUniformComponents / 4)"},
        {"gl_MaxFragmentUniformVectors", "(gl_MaxFragmentUniformComponents / 4)"},
    };
    std::string out = "#version 120\n";
    size_t lineStart = 0;
    while (lineStart < preprocessed.size()) {
        size_t lineEnd = preprocessed.find('\n', lineStart);
        if (lineEnd == std::string::npos) lineEnd = preprocessed.size();
        std::string line = preprocessed.substr(lineStart, lineEnd - lineStart);
        lineStart = lineEnd + 1;
        const size_t first = line.find_first_not_of(" \t");
        if (first != std::string::npos && line[first] == '#') {
            // #version and #extension lines go (blank: line numbers stay).
            const size_t word = line.find_first_not_of(" \t", first + 1);
            if (word != std::string::npos && (line.compare(word, 7, "version") == 0 || line.compare(word, 9, "extension") == 0)) {
                out += '\n';
                continue;
            }
            out += line + '\n';
            continue;
        }
        for (size_t i = 0; i < line.size();) {
            if (std::isalpha(static_cast<unsigned char>(line[i])) || line[i] == '_') {
                size_t end = i;
                while (end < line.size() && (std::isalnum(static_cast<unsigned char>(line[end])) || line[end] == '_')) ++end;
                const auto it = kRename.find(line.substr(i, end - i));
                out += it == kRename.end() ? line.substr(i, end - i) : it->second;
                i = end;
            } else if (std::isdigit(static_cast<unsigned char>(line[i]))) {
                size_t end = i;
                while (end < line.size() && (std::isalnum(static_cast<unsigned char>(line[end])) || line[end] == '_' || line[end] == '.')) ++end;
                out += line.substr(i, end - i);
                i = end;
            } else {
                out += line[i++];
            }
        }
        out += '\n';
    }
    return out;
}

std::string relaxDesktopGlsl(const std::string& source) {
    const GlslVersion version = parseGlslVersion(source);
    if (version.es) return source;
    // Built-ins a shader may redeclare plainly ("<storage> <type> <name>;")
    // where glslang refuses it.
    struct Redeclaration {
        const char* name;
        const char* storage;
        const char* type;
        bool array;
    };
    static const Redeclaration kRedeclarations[] = {
        {"gl_Position", "out", "vec4", false}, {"gl_PointSize", "out", "float", false},
        {"gl_FragColor", "out", "vec4", false}, {"gl_FragData", "out", "vec4", true},
        {"gl_FragDepth", "out", "float", false}, {"gl_FragCoord", "in", "vec4", false},
    };
    auto compatibilityBuiltin = [](const std::string& word) {
        if (word.rfind("gl_", 0) != 0) return false;
        for (const BuiltinDecl& b : kVertexInputs)
            if (word == b.name) return true;
        for (const BuiltinDecl& b : kVaryings)
            if (word == b.name) return true;
        for (const char* u : kFixedFunctionUniforms)
            if (word == u) return true;
        return word == "gl_TexCoord" || word == "gl_FragColor" || word == "gl_FragData";
    };

    // Tokens outside comments and preprocessor directives.
    struct Token {
        size_t begin, end;
    };
    std::vector<Token> tokens;
    size_t versionEnd = std::string::npos;  // after the #version directive's number
    const size_t n = source.size();
    bool lineStart = true;
    for (size_t i = 0; i < n;) {
        const char c = source[i];
        if (c == '\n') {
            lineStart = true;
            ++i;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
        } else if (source.compare(i, 2, "//") == 0) {
            while (i < n && source[i] != '\n') ++i;
        } else if (source.compare(i, 2, "/*") == 0) {
            const size_t end = source.find("*/", i + 2);
            i = end == std::string::npos ? n : end + 2;
        } else if (c == '#' && lineStart) {
            size_t j = i + 1;
            while (j < n && (source[j] == ' ' || source[j] == '\t')) ++j;
            if (versionEnd == std::string::npos && source.compare(j, 7, "version") == 0) {
                j += 7;
                while (j < n && (source[j] == ' ' || source[j] == '\t')) ++j;
                while (j < n && std::isdigit(static_cast<unsigned char>(source[j]))) ++j;
                versionEnd = j;
            }
            while (i < n && source[i] != '\n') i += source[i] == '\\' && i + 1 < n && source[i + 1] == '\n' ? 2 : 1;
        } else {
            lineStart = false;
            size_t end = i + 1;
            if (isIdent(c))
                while (end < n && isIdent(source[end])) ++end;
            tokens.push_back({i, end});
            i = end;
        }
    }
    auto text = [&](size_t t) {
        return t < tokens.size() ? source.substr(tokens[t].begin, tokens[t].end - tokens[t].begin) : std::string();
    };

    std::string out = source;
    bool usesCompatibilityBuiltins = false;
    int depth = 0;
    for (size_t t = 0; t < tokens.size(); ++t) {
        const std::string word = text(t);
        depth += word == "{" ? 1 : word == "}" ? -1 : 0;
        usesCompatibilityBuiltins |= compatibilityBuiltin(word);
        if (depth != 0 || (t > 0 && text(t - 1) != ";" && text(t - 1) != "}")) continue;
        const std::string name = text(t + 2);
        const Redeclaration* r = nullptr;
        for (const Redeclaration& candidate : kRedeclarations)
            if (name == candidate.name && word == candidate.storage && text(t + 1) == candidate.type) r = &candidate;
        if (!r) continue;
        size_t last = t + 3;
        if (r->array && text(last) == "[") {
            ++last;
            if (!text(last).empty() && std::isdigit(static_cast<unsigned char>(text(last)[0]))) ++last;
            if (text(last) != "]") continue;
            ++last;
        }
        if (text(last) != ";") continue;
        for (size_t k = tokens[t].begin; k < tokens[last].end; ++k)
            if (out[k] != '\n') out[k] = ' ';
        usesCompatibilityBuiltins |= compatibilityBuiltin(name);
        t = last;
    }
    if (versionEnd != std::string::npos && version.number >= 150 && !version.compatibility && !version.core &&
        usesCompatibilityBuiltins)
        out.insert(versionEnd, " compatibility");
    return out;
}

UpgradeResult upgradeLegacyGlsl(Stage stage, const std::string& source, const UpgradeOptions& options) {
    UpgradeResult result;
    const GlslVersion version = parseGlslVersion(source);
    if (!version.compatibility && version.number >= 140) {
        result.source = source;
        return result;
    }
    const int stageBit = 1 << static_cast<int>(stage);
    const bool vertex = stage == Stage::Vertex, fragment = stage == Stage::Fragment;
    const int newVersion = version.number < 140 ? 140 : version.number;

    std::map<std::string, std::string> rename;
    if (vertex) {
        rename["attribute"] = "in";
        rename["varying"] = "out";
    } else if (fragment) {
        rename["varying"] = "in";
    } else if (stage == Stage::Geometry) {
        rename["varying"] = "out";
    }
    // __VERSION__ stays the shader's own: its #if __VERSION__ >= 130 blocks
    // choose what they would on a GL driver, not the GLSL 1.40 the upgrade
    // writes (osgText's fragment shader, with no #version, read glyph
    // coverage from .r instead of .a and drew nothing, issue #42).
    const std::string ownVersion = std::to_string(version.number);
    if (version.number != newVersion) rename["__VERSION__"] = ownVersion;

    // GLSL 1.x texture functions. Under the GLSL 1.40+ rules used for codegen
    // texture2D etc. are type names, so calls must use the generic functions.
    static const std::map<std::string, std::string> kTextureFunctions = {
        {"texture1D", "texture"}, {"texture2D", "texture"}, {"texture3D", "texture"}, {"textureCube", "texture"},
        {"texture2DRect", "texture"}, {"texture1DProj", "textureProj"}, {"texture2DProj", "textureProj"},
        {"texture3DProj", "textureProj"}, {"texture2DRectProj", "textureProj"}, {"texture1DLod", "textureLod"},
        {"texture2DLod", "textureLod"}, {"texture3DLod", "textureLod"}, {"textureCubeLod", "textureLod"},
        {"texture1DProjLod", "textureProjLod"}, {"texture2DProjLod", "textureProjLod"},
        {"texture3DProjLod", "textureProjLod"}, {"texture1DGradARB", "textureGrad"}, {"texture2DGradARB", "textureGrad"},
        {"texture3DGradARB", "textureGrad"}, {"textureCubeGradARB", "textureGrad"},
    };
    for (const auto& [from, to] : kTextureFunctions) rename[from] = to;
    static const std::set<std::string> kGenericTextureFunctions = {"texture", "textureProj", "textureLod",
                                                                   "textureProjLod", "textureGrad"};
    // shadow*() returned vec4 in GLSL 1.x; texture() on shadow samplers
    // returns float: the call becomes vec4(texture(...)). (Inline, not a
    // helper function: several shader objects of a stage may use them.)
    static const std::map<std::string, std::string> kShadowFunctions = {
        {"shadow1D", "texture"}, {"shadow2D", "texture"}, {"shadow2DRect", "texture"},
        {"shadow1DProj", "textureProj"}, {"shadow2DProj", "textureProj"}, {"shadow2DRectProj", "textureProj"},
        {"shadow1DLod", "textureLod"}, {"shadow2DLod", "textureLod"},
        {"shadow1DProjLod", "textureProjLod"}, {"shadow2DProjLod", "textureProjLod"},
    };
    // Built-in varyings a shader may redeclare (GLSL 1.10 sizes gl_TexCoord,
    // 1.30 adds interpolation qualifiers), by the key `used` records them under.
    static const std::map<std::string, std::string> kVertexRedeclarable = {
        {"gl_FrontColor", "gl_FrontColor"}, {"gl_BackColor", "gl_BackColor"},
        {"gl_FrontSecondaryColor", "gl_FrontSecondaryColor"}, {"gl_BackSecondaryColor", "gl_BackSecondaryColor"},
        {"gl_FogFragCoord", "gl_FogFragCoord"}, {"gl_TexCoord", "gl_TexCoord"},
    };
    static const std::map<std::string, std::string> kFragmentRedeclarable = {
        {"gl_Color", "fs:FrontColor"}, {"gl_SecondaryColor", "fs:FrontSecondaryColor"},
        {"gl_TexCoord", "fs:TexCoord"}, {"gl_FogFragCoord", "fs:FogFragCoord"},
    };

    std::set<std::string> used;
    std::set<std::string> redeclared;  // built-in varyings the shader declares itself (keys of `used`)
    bool hasMain = false;
    std::string out;
    out.reserve(source.size() + 512);
    const size_t n = source.size();
    size_t i = 0;
    bool versionReplaced = false;
    bool lineStart = true;
    size_t insertAt = std::string::npos;  // after the leading directive block
    // Declarations go before the first token, outside any #if block (which
    // may be compiled out, as in "#ifdef GL_ES precision ... #endif").
    int conditionalDepth = 0;
    size_t conditionalStart = 0;
    int braceDepth = 0, parenDepth = 0;
    std::vector<int> shadowCalls;  // paren depth of each open shadow*() call
    std::string previousWord;      // the identifier just emitted ("" after other tokens)
    size_t defineEnd = 0;          // end of the #define line whose body is being rewritten
    while (i < n) {
        const char c = source[i];
        const bool inDefine = i < defineEnd;
        if (inDefine && c == '\\' && i + 1 < n && source[i + 1] == '\n') {
            out += "\\\n";  // a continued macro body
            i += 2;
            continue;
        }
        if (c == '\n') {
            out += c;
            ++i;
            lineStart = true;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            out += c;
            ++i;
            continue;
        }
        if (source.compare(i, 2, "//") == 0) {
            while (i < n && source[i] != '\n') out += source[i++];
            continue;
        }
        if (source.compare(i, 2, "/*") == 0) {
            const size_t end = source.find("*/", i + 2);
            const size_t stop = end == std::string::npos ? n : end + 2;
            out.append(source, i, stop - i);
            i = stop;
            continue;
        }
        if (c == '#' && lineStart) {
            size_t lineEnd = i;
            while (lineEnd < n && source[lineEnd] != '\n') {
                if (source[lineEnd] == '\\' && lineEnd + 1 < n && source[lineEnd + 1] == '\n') lineEnd += 2;
                else ++lineEnd;
            }
            std::string directive = source.substr(i, lineEnd - i);
            size_t k = 1;
            while (k < directive.size() && (directive[k] == ' ' || directive[k] == '\t')) ++k;
            size_t nameEnd = k;
            while (nameEnd < directive.size() && isIdent(directive[nameEnd])) ++nameEnd;
            const std::string name = directive.substr(k, nameEnd - k);
            if (name == "if" || name == "ifdef" || name == "ifndef") {
                if (conditionalDepth++ == 0) conditionalStart = out.size();
            } else if (name == "endif" && conditionalDepth > 0) {
                --conditionalDepth;
            }
            if (name == "define") {
                // Macro bodies get the same rewriting as code (renamed built-ins
                // and functions): only the macro's name and parameters are kept.
                size_t head = nameEnd;
                while (head < directive.size() && (directive[head] == ' ' || directive[head] == '\t')) ++head;
                while (head < directive.size() && isIdent(directive[head])) ++head;
                if (head < directive.size() && directive[head] == '(') {
                    const size_t close = directive.find(')', head);
                    head = close == std::string::npos ? directive.size() : close + 1;
                }
                out.append(directive, 0, head);
                i += head;
                defineEnd = lineEnd;
                lineStart = false;
                continue;
            }
            if (!versionReplaced && directive.compare(k, 7, "version") == 0) {
                out += "#version " + std::to_string(newVersion);
                versionReplaced = true;
            } else if (version.number != newVersion) {
                out += replaceIdentifier(directive, "__VERSION__", ownVersion);
            } else {
                out += directive;
            }
            i = lineEnd;
            lineStart = false;
            continue;
        }
        if (insertAt == std::string::npos && !inDefine) {
            // First real token: declarations go before it (or before the
            // #if block holding it), on their own line.
            insertAt = conditionalDepth > 0 ? conditionalStart : out.size();
        }
        lineStart = false;
        if (isIdentStart(c)) {
            size_t end = i;
            while (end < n && isIdent(source[end])) ++end;
            std::string word = source.substr(i, end - i);
            i = end;
            const std::string previous = previousWord;
            previousWord = word;
            // The generic functions the texture renames call are ordinary
            // names in GLSL 1.10-1.20: a variable named `texture` would hide
            // the function (Another Brick in the Mall's
            // "uniform sampler2D texture;" read with texture2D(texture, ...),
            // issue #100). Names not called become escaped user names
            // (kEscapePrefix, which the program interface strips again).
            // Macro bodies keep them: "#define TEX texture" names the function.
            if (!inDefine && kGenericTextureFunctions.count(word)) {
                size_t next = i;
                while (next < n && std::isspace(static_cast<unsigned char>(source[next]))) ++next;
                if (next >= n || source[next] != '(') {
                    out += kEscapePrefix + word;
                    continue;
                }
            }
            auto it = rename.find(word);
            if (it != rename.end()) {
                out += it->second;
                continue;
            }
            if (auto shadow = kShadowFunctions.find(word); shadow != kShadowFunctions.end()) {
                out += "vec4(" + shadow->second;
                shadowCalls.push_back(parenDepth);
                continue;
            }
            if (word.rfind("gl_", 0) == 0 && braceDepth == 0 && (previous == "vec4" || previous == "float")) {
                // A redeclared built-in varying: the shader's declaration
                // (with its qualifiers) replaces the generated one.
                const auto& redeclarable = vertex ? kVertexRedeclarable : kFragmentRedeclarable;
                if (auto r = redeclarable.find(word); (vertex || fragment) && r != redeclarable.end()) {
                    redeclared.insert(r->second);
                    if (r->second.compare(0, 3, "fs:") == 0) {
                        result.builtinVaryingsRead.insert(r->second.substr(3));
                        used.insert(r->second);
                        out += "wmgl_" + r->second.substr(3);
                    } else {
                        used.insert(word);
                        out += "wmgl_" + word.substr(3);
                    }
                    if (word == "gl_TexCoord") {
                        // Every stage declares all 8 so the interfaces match.
                        size_t j = i;
                        while (j < n && std::isspace(static_cast<unsigned char>(source[j]))) ++j;
                        if (j < n && source[j] == '[') {
                            const size_t close = source.find(']', j);
                            i = close == std::string::npos ? n : close + 1;
                        }
                        out += "[8]";
                    }
                    previousWord.clear();
                    continue;
                }
            }
            if (word.rfind("gl_", 0) == 0) {
                if (fragment && (word == "gl_FragColor" || word == "gl_FragData")) {
                    used.insert(word);
                    out += "wmgl_" + word.substr(3);
                    continue;
                }
                if (fragment && (word == "gl_Color" || word == "gl_SecondaryColor" || word == "gl_TexCoord" ||
                                 word == "gl_FogFragCoord")) {
                    // gl_Color and gl_SecondaryColor read the front-facing vertex colours.
                    const std::string varying = word == "gl_Color"            ? "FrontColor"
                                                : word == "gl_SecondaryColor" ? "FrontSecondaryColor"
                                                                              : word.substr(3);
                    result.builtinVaryingsRead.insert(varying);
                    used.insert("fs:" + varying);
                    out += "wmgl_" + varying;
                    continue;
                }
                bool handled = false;
                for (const BuiltinDecl& b : kVertexInputs)
                    if ((b.stages & stageBit) && word == b.name) handled = true;
                for (const BuiltinDecl& b : kVaryings)
                    if ((b.stages & stageBit) && word == b.name) handled = true;
                if (vertex && word == "gl_TexCoord") handled = true;
                for (const char* u : kFixedFunctionUniforms)
                    if (word == u) {
                        handled = true;
                        result.usesFixedFunctionState = true;
                    }
                if (handled) {
                    used.insert(word);
                    out += "wmgl_" + word.substr(3);
                    continue;
                }
            }
            if ((vertex || fragment) && word == "main") {
                hasMain = true;
                out += "wmgl_user_main";
                continue;
            }
            if (vertex && word == "ftransform") {
                used.insert("ftransform");
                used.insert("gl_Vertex");
                result.usesFixedFunctionState = true;
                out += "wmgl_ftransform";
                continue;
            }
            out += word;
            continue;
        }
        previousWord.clear();
        braceDepth += c == '{' ? 1 : c == '}' ? -1 : 0;
        out += c;
        ++i;
        if (c == '(') ++parenDepth;
        if (c == ')' && --parenDepth >= 0 && !shadowCalls.empty() && parenDepth == shadowCalls.back()) {
            out += ')';  // closes vec4( of a shadow*() call
            shadowCalls.pop_back();
        }
    }

    // Declarations for everything that was renamed.
    std::string decls = "\n";
    if (used.count("gl_FragColor") && used.count("gl_FragData")) {
        result.ok = false;
        result.log = "ERROR: a fragment shader cannot write both gl_FragColor and gl_FragData\n";
        return result;
    }
    if (used.count("gl_FragColor")) decls += "out vec4 wmgl_FragColor;\n";
    if (used.count("gl_FragData")) decls += "out vec4 wmgl_FragData[8];\n";
    for (const BuiltinDecl& b : kVertexInputs)
        if ((b.stages & stageBit) && used.count(b.name)) decls += std::string(b.decl) + "\n";
    for (const BuiltinDecl& b : kVaryings)
        if ((b.stages & stageBit) && used.count(b.name) && !redeclared.count(b.name)) decls += std::string(b.decl) + "\n";
    if (vertex && used.count("gl_TexCoord") && !redeclared.count("gl_TexCoord")) decls += "out vec4 wmgl_TexCoord[8];\n";
    if (fragment) {
        auto declare = [&](const char* key, const char* decl) {
            if (used.count(key) && !redeclared.count(key)) decls += decl;
        };
        declare("fs:FrontColor", "in vec4 wmgl_FrontColor;\n");
        declare("fs:FrontSecondaryColor", "in vec4 wmgl_FrontSecondaryColor;\n");
        declare("fs:TexCoord", "in vec4 wmgl_TexCoord[8];\n");
        declare("fs:FogFragCoord", "in float wmgl_FogFragCoord;\n");
    }
    // The generated main (see glsl_upgrade.h).
    std::string prologue, epilogue;
    if (vertex) {
        for (const std::string& varying : options.provideVaryings) {
            if (used.count("gl_" + varying)) continue;
            if (varying == "TexCoord") {
                decls += "out vec4 wmgl_TexCoord[8];\n";
                prologue += "    for (int i = 0; i < 8; ++i) wmgl_TexCoord[i] = vec4(0.0);\n";
            } else if (varying == "FogFragCoord") {
                decls += "out float wmgl_FogFragCoord;\n";
                prologue += "    wmgl_FogFragCoord = 0.0;\n";
            } else if (varying == "FrontColor" || varying == "FrontSecondaryColor") {
                decls += "out vec4 wmgl_" + varying + ";\n";
                prologue += "    wmgl_" + varying + " = vec4(0.0);\n";
            }
        }
        if (used.count("gl_ClipVertex")) {
            result.usesFixedFunctionState = true;
            for (int i = 0; i < 8; ++i)
                epilogue += "    gl_ClipDistance[" + std::to_string(i) + "] = dot(wmgl_ClipVertex, wmgl_ClipPlane[" +
                            std::to_string(i) + "]);\n";
        }
    }
    if (fragment && (used.count("gl_FragColor") || used.count("gl_FragData"))) {
        result.usesFixedFunctionState = true;
        epilogue += std::string("    wmgl_alphaTest(") + (used.count("gl_FragColor") ? "wmgl_FragColor" : "wmgl_FragData[0]") +
                    ".a);\n";
    }
    if (result.usesFixedFunctionState) {
        // Fixed-function state, fed by the compatibility profile.
        decls +=
            "struct wmgl_LightSourceParameters { vec4 ambient; vec4 diffuse; vec4 specular; vec4 position;"
            " vec4 halfVector; vec3 spotDirection; float spotExponent; float spotCutoff; float spotCosCutoff;"
            " float constantAttenuation; float linearAttenuation; float quadraticAttenuation; };\n"
            "struct wmgl_MaterialParameters { vec4 emission; vec4 ambient; vec4 diffuse; vec4 specular; float shininess; };\n"
            "struct wmgl_LightModelParameters { vec4 ambient; };\n"
            "struct wmgl_LightModelProducts { vec4 sceneColor; };\n"
            "struct wmgl_LightProducts { vec4 ambient; vec4 diffuse; vec4 specular; };\n"
            "struct wmgl_FogParameters { vec4 color; float density; float start; float end; float scale; };\n"
            "struct wmgl_PointParameters { float size; float sizeMin; float sizeMax; float fadeThresholdSize;"
            " float distanceConstantAttenuation; float distanceLinearAttenuation; float distanceQuadraticAttenuation; };\n"
            "layout(std140) uniform wmgl_FixedFunction {\n"
            "  mat4 wmgl_ModelViewMatrix; mat4 wmgl_ProjectionMatrix; mat4 wmgl_ModelViewProjectionMatrix;\n"
            "  mat4 wmgl_TextureMatrix[8]; mat3 wmgl_NormalMatrix;\n"
            "  mat4 wmgl_ModelViewMatrixInverse; mat4 wmgl_ProjectionMatrixInverse; mat4 wmgl_ModelViewProjectionMatrixInverse;\n"
            "  mat4 wmgl_TextureMatrixInverse[8];\n"
            "  mat4 wmgl_ModelViewMatrixTranspose; mat4 wmgl_ProjectionMatrixTranspose; mat4 wmgl_ModelViewProjectionMatrixTranspose;\n"
            "  mat4 wmgl_TextureMatrixTranspose[8];\n"
            "  mat4 wmgl_ModelViewMatrixInverseTranspose; mat4 wmgl_ProjectionMatrixInverseTranspose;\n"
            "  mat4 wmgl_ModelViewProjectionMatrixInverseTranspose; mat4 wmgl_TextureMatrixInverseTranspose[8];\n"
            "  float wmgl_NormalScale; vec4 wmgl_ClipPlane[8]; wmgl_PointParameters wmgl_Point;\n"
            "  wmgl_MaterialParameters wmgl_FrontMaterial; wmgl_MaterialParameters wmgl_BackMaterial;\n"
            "  wmgl_LightSourceParameters wmgl_LightSource[8]; wmgl_LightModelParameters wmgl_LightModel;\n"
            "  wmgl_LightModelProducts wmgl_FrontLightModelProduct; wmgl_LightModelProducts wmgl_BackLightModelProduct;\n"
            "  wmgl_LightProducts wmgl_FrontLightProduct[8]; wmgl_LightProducts wmgl_BackLightProduct[8];\n"
            "  vec4 wmgl_TextureEnvColor[8]; vec4 wmgl_EyePlaneS[8]; vec4 wmgl_EyePlaneT[8]; vec4 wmgl_EyePlaneR[8];\n"
            "  vec4 wmgl_EyePlaneQ[8]; vec4 wmgl_ObjectPlaneS[8]; vec4 wmgl_ObjectPlaneT[8]; vec4 wmgl_ObjectPlaneR[8];\n"
            "  vec4 wmgl_ObjectPlaneQ[8]; wmgl_FogParameters wmgl_Fog;\n"
            "  vec4 wmgl_AlphaTest;\n"
            "  vec4 wmgl_LineParams;\n"
            "  uvec4 wmgl_Stipple[8];\n"
            "};\n";
        // Only the object with main calls it (from the generated main), and
        // a stage may have several objects.
        if (fragment && hasMain)
            decls +=
                "void wmgl_alphaTest(float a) {\n"
                "  int f = int(wmgl_AlphaTest.x); float r = wmgl_AlphaTest.y;\n"
                "  if (f != 0 && !(f == 8 || (f == 2 && a < r) || (f == 3 && a == r) || (f == 4 && a <= r) ||\n"
                "      (f == 5 && a > r) || (f == 6 && a != r) || (f == 7 && a >= r))) discard;\n"
                "}\n";
        if (vertex && used.count("ftransform"))
            decls += "#define wmgl_ftransform() (wmgl_ModelViewProjectionMatrix * wmgl_Vertex)\n";
        if (vertex && used.count("ftransform") && !used.count("gl_Vertex")) decls += "in vec4 wmgl_Vertex;\n";
    }
    if (decls.size() > 1) {
        if (insertAt == std::string::npos) insertAt = out.size();
        out.insert(insertAt, decls);
        // Keep line numbers stable for error messages after the declarations.
        out.insert(insertAt + decls.size(), "#line " + std::to_string(1 + std::count(out.begin(), out.begin() + static_cast<long>(insertAt), '\n')) + "\n");
    }
    if (!versionReplaced) out = "#version " + std::to_string(newVersion) + "\n#line 1\n" + out;
    if (hasMain) {
        // A shader object without main (one of several per stage) gets no wrapper.
        out += "\nvoid main() {\n" + prologue + "    wmgl_user_main();\n" + epilogue + "}\n";
    }
    result.source = out;
    return result;
}

namespace {

const std::set<std::string>& vulkanOnlyKeywords() {
    static const std::set<std::string> words = [] {
        std::set<std::string> w = {"sampler", "samplerShadow", "subpassInput", "isubpassInput", "usubpassInput",
                                   "subpassInputMS", "isubpassInputMS", "usubpassInputMS"};
        for (const char* prefix : {"", "i", "u"})
            for (const char* dim : {"1D", "1DArray", "2D", "2DArray", "2DRect", "3D", "Cube", "CubeArray", "Buffer",
                                    "2DMS", "2DMSArray"})
                w.insert(std::string(prefix) + "texture" + dim);
        return w;
    }();
    return words;
}

}  // namespace

std::string escapeVulkanKeywords(const std::string& source) {
    const std::set<std::string>& words = vulkanOnlyKeywords();
    std::string out;
    out.reserve(source.size());
    const size_t n = source.size();
    size_t i = 0;
    while (i < n) {
        if (source.compare(i, 2, "//") == 0) {
            const size_t end = source.find('\n', i);
            const size_t stop = end == std::string::npos ? n : end;
            out.append(source, i, stop - i);
            i = stop;
            continue;
        }
        if (source.compare(i, 2, "/*") == 0) {
            const size_t end = source.find("*/", i + 2);
            const size_t stop = end == std::string::npos ? n : end + 2;
            out.append(source, i, stop - i);
            i = stop;
            continue;
        }
        if (isIdentStart(source[i])) {
            size_t end = i;
            while (end < n && isIdent(source[end])) ++end;
            const std::string word = source.substr(i, end - i);
            if (words.count(word)) out += kEscapePrefix;
            out += word;
            i = end;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(source[i]))) {
            // Numbers can contain identifier characters (1e5, 0x1Fu); copy them whole.
            while (i < n && isIdent(source[i])) out += source[i++];
            continue;
        }
        out += source[i++];
    }
    return out;
}

std::string unescapeName(std::string name) {
    const std::string prefix = kEscapePrefix;
    for (size_t at = name.find(prefix); at != std::string::npos; at = name.find(prefix, at)) name.erase(at, prefix.size());
    return name;
}

std::string spliceLineContinuations(const std::string& source) {
    if (source.find('\\') == std::string::npos) return source;
    std::string out;
    out.reserve(source.size());
    int joined = 0;  // newlines removed from the current line
    for (size_t i = 0; i < source.size(); ++i) {
        const char c = source[i];
        if (c == '\\') {
            size_t j = i + 1;
            if (j < source.size() && source[j] == '\r') ++j;
            if (j < source.size() && source[j] == '\n') {
                ++joined;
                i = j;
                continue;
            }
        }
        out += c;
        if (c == '\n' && joined) {
            out.append(static_cast<size_t>(joined), '\n');
            joined = 0;
        }
    }
    out.append(static_cast<size_t>(joined), '\n');
    return out;
}

std::string checkReservedWords(const std::string& source) {
    const GlslVersion version = parseGlslVersion(source);
    if (version.es || version.number < 130 || version.number >= 440) return "";
    // The reserved words of each GLSL version's spec (section 3.6) that
    // glslang accepts as identifiers. GLSL 4.00 makes the double types
    // keywords; 4.20 makes the image types and volatile keywords and
    // reserves resource; 4.40 frees packed and row_major.
    static const std::set<std::string> common = {
        "common", "partition", "active", "asm", "class", "union", "enum", "typedef", "template", "this", "packed",
        "goto", "inline", "noinline", "public", "static", "extern", "external", "interface", "long", "short", "half",
        "fixed", "unsigned", "superp", "input", "output", "hvec2", "hvec3", "hvec4", "fvec2", "fvec3", "fvec4",
        "sampler3DRect", "filter", "sizeof", "cast", "namespace", "using", "row_major"};
    static const std::set<std::string> before420 = {
        "volatile", "image1D", "image2D", "image3D", "imageCube", "iimage1D", "iimage2D", "iimage3D", "iimageCube",
        "uimage1D", "uimage2D", "uimage3D", "uimageCube", "image1DArray", "image2DArray", "iimage1DArray",
        "iimage2DArray", "uimage1DArray", "uimage2DArray", "image1DShadow", "image2DShadow", "image1DArrayShadow",
        "image2DArrayShadow", "imageBuffer", "iimageBuffer", "uimageBuffer"};
    static const std::set<std::string> before400 = {"double", "dvec2", "dvec3", "dvec4"};
    // Extensions that make some of them keywords early.
    auto enables = [&](const char* extension) {
        for (size_t at = source.find(extension); at != std::string::npos; at = source.find(extension, at + 1)) {
            const size_t line = source.rfind('\n', at);
            const size_t start = line == std::string::npos ? 0 : line + 1;
            const size_t hash = source.find_first_not_of(" \t", start);
            if (hash != std::string::npos && source.compare(hash, 10, "#extension") == 0) return true;
        }
        return false;
    };
    std::set<std::string> reserved = common;
    if (version.number < 420 && !enables("GL_ARB_shader_image_load_store")) reserved.insert(before420.begin(), before420.end());
    if (version.number >= 420) reserved.insert("resource");
    if (version.number < 400 && !enables("GL_ARB_gpu_shader_fp64")) reserved.insert(before400.begin(), before400.end());
    std::string errors;
    const size_t n = source.size();
    size_t i = 0;
    int line = 1;
    bool lineStart = true;
    int layoutDepth = -1;  // parenthesis depth inside layout(...), -1 outside
    int depth = 0;
    std::string previous;
    while (i < n) {
        const char c = source[i];
        if (c == '\n') {
            ++line;
            ++i;
            lineStart = true;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }
        if (source.compare(i, 2, "//") == 0) {
            while (i < n && source[i] != '\n') ++i;
            continue;
        }
        if (source.compare(i, 2, "/*") == 0) {
            const size_t end = source.find("*/", i + 2);
            const size_t stop = end == std::string::npos ? n : end + 2;
            for (size_t k = i; k < stop; ++k) line += source[k] == '\n';
            i = stop;
            continue;
        }
        if (c == '#' && lineStart) {
            while (i < n && source[i] != '\n') {
                if (source[i] == '\\' && i + 1 < n && source[i + 1] == '\n') ++line, ++i;
                ++i;
            }
            continue;
        }
        lineStart = false;
        if (isIdentStart(c)) {
            size_t end = i;
            while (end < n && isIdent(source[end])) ++end;
            const std::string word = source.substr(i, end - i);
            i = end;
            if (layoutDepth < 0 && reserved.count(word))
                errors += "ERROR: 0:" + std::to_string(line) + ": '" + word + "' : Reserved word.\n";
            previous = word;
            continue;
        }
        if (c == '(') {
            if (previous == "layout" && layoutDepth < 0) layoutDepth = depth;
            ++depth;
        } else if (c == ')') {
            --depth;
            if (depth == layoutDepth) layoutDepth = -1;
        }
        previous.clear();
        ++i;
    }
    return errors;
}

std::string lowerBuiltinUniforms(Stage stage, const std::string& source) {
    const bool fragment = stage == Stage::Fragment;
    // Vertex shaders' draw parameters (GL 4.6 names, or the ARB ones with the extension).
    bool drawParameters = false, drawParametersArb = false;
    if (stage == Stage::Vertex && (source.find("gl_BaseVertex") != std::string::npos ||
                                   source.find("gl_BaseInstance") != std::string::npos ||
                                   source.find("gl_DrawID") != std::string::npos)) {
        static const std::regex kVersion(R"(#[ \t]*version[ \t]+([0-9]+))");
        std::smatch m;
        // (A version too long for an int is not one glslang takes either.)
        drawParameters = std::regex_search(source, m, kVersion) && m[1].length() <= 4 && std::stoi(m[1]) >= 460;
        drawParametersArb = source.find("GL_ARB_shader_draw_parameters") != std::string::npos;
    }
    if (source.find("gl_DepthRange") == std::string::npos &&
        (!fragment || source.find("gl_NumSamples") == std::string::npos) && !drawParameters && !drawParametersArb)
        return source;
    const std::string prefix = kBuiltinUniformPrefix;
    std::string out;
    out.reserve(source.size() + 256);
    bool depthRange = false, numSamples = false, lineStart = true;
    bool inDefine = false;  // in a #define line: its body is rewritten, but declarations never go there
    std::set<std::string> parameters;  // draw parameters used: "BaseVertex", "BaseInstance", "DrawID"
    size_t insertAt = std::string::npos;  // before the first token, outside any #if block
    int conditionalDepth = 0;
    size_t conditionalStart = 0;
    const size_t n = source.size();
    size_t i = 0;
    while (i < n) {
        const char c = source[i];
        if (c == '\n') {
            out += c;
            ++i;
            lineStart = true;
            inDefine = false;
            continue;
        }
        if (c == '\\' && i + 1 < n && source[i + 1] == '\n') {  // a continued line
            out.append(source, i, 2);
            i += 2;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c))) {
            out += c;
            ++i;
            continue;
        }
        if (source.compare(i, 2, "//") == 0) {
            while (i < n && source[i] != '\n') out += source[i++];
            continue;
        }
        if (source.compare(i, 2, "/*") == 0) {
            const size_t end = source.find("*/", i + 2);
            const size_t stop = end == std::string::npos ? n : end + 2;
            out.append(source, i, stop - i);
            i = stop;
            continue;
        }
        if (c == '#' && lineStart) {
            size_t lineEnd = i;
            while (lineEnd < n && source[lineEnd] != '\n') {
                if (source[lineEnd] == '\\' && lineEnd + 1 < n && source[lineEnd + 1] == '\n') lineEnd += 2;
                else ++lineEnd;
            }
            size_t k = i + 1;
            while (k < lineEnd && (source[k] == ' ' || source[k] == '\t')) ++k;
            size_t nameEnd = k;
            while (nameEnd < lineEnd && isIdent(source[nameEnd])) ++nameEnd;
            const std::string name = source.substr(k, nameEnd - k);
            if (name == "if" || name == "ifdef" || name == "ifndef") {
                if (conditionalDepth++ == 0) conditionalStart = out.size();
            } else if (name == "endif" && conditionalDepth > 0) {
                --conditionalDepth;
            }
            if (name != "define") {
                // Other directives are copied; #define bodies are rewritten like code.
                out.append(source, i, lineEnd - i);
                i = lineEnd;
                continue;
            }
            out.append(source, i, nameEnd - i);
            i = nameEnd;
            lineStart = false;
            inDefine = true;
            continue;
        }
        if (insertAt == std::string::npos && !inDefine && conditionalDepth == 0) insertAt = out.size();
        if (insertAt == std::string::npos && !inDefine && conditionalDepth > 0) insertAt = conditionalStart;
        lineStart = false;
        if (isIdentStart(c)) {
            size_t end = i;
            while (end < n && isIdent(source[end])) ++end;
            const std::string word = source.substr(i, end - i);
            i = end;
            if (word == "gl_DepthRange" || word == "gl_DepthRangeParameters") {
                depthRange = true;
                out += prefix + word.substr(3);
            } else if (fragment && word == "gl_NumSamples") {
                numSamples = true;
                out += prefix + word.substr(3);
            } else if ((drawParameters && (word == "gl_BaseVertex" || word == "gl_BaseInstance" || word == "gl_DrawID")) ||
                       (drawParametersArb &&
                        (word == "gl_BaseVertexARB" || word == "gl_BaseInstanceARB" || word == "gl_DrawIDARB"))) {
                const std::string name = word.substr(3, word.size() - 3 - (word.back() == 'B' ? 3 : 0));
                parameters.insert(name);
                out += prefix + name;
            } else {
                out += word;
            }
            continue;
        }
        out += c;
        ++i;
    }
    std::string decls;
    if (depthRange)
        decls += "struct " + prefix + "DepthRangeParameters { float near; float far; float diff; };\n"
                 "uniform " + prefix + "DepthRangeParameters " + prefix + "DepthRange;\n";
    if (numSamples) decls += "uniform int " + prefix + "NumSamples;\n";
    for (const std::string& name : parameters) decls += "uniform int " + prefix + name + ";\n";
    if (decls.empty()) return source;
    if (insertAt == std::string::npos) insertAt = out.size();
    const long line = 1 + std::count(out.begin(), out.begin() + static_cast<long>(insertAt), '\n');
    out.insert(insertAt, decls + "#line " + std::to_string(line) + "\n");
    return out;
}

}  // namespace wmgl::shader
