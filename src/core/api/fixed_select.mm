// Selection and feedback (GL 3.3 compatibility sections 5.2 and 5.3).
//
// The render modes and the name stack are tracked, but no hit or feedback
// records are produced: outside GL_RENDER nothing is drawn and glRenderMode
// returns 0 records. Picking code finds no hits instead of failing.

#include "core/compat/compat_api.h"
#include "core/compat/fixed_state.h"
#include "core/context.h"
#include "core/util/log.h"
#include "gl_api.gen.h"

namespace wmgl::api {
namespace {

constexpr size_t kMaxNameStackDepth = 64;  // GL_MAX_NAME_STACK_DEPTH

bool compatOnly(Context& ctx) {
    if (ctx.isCompat()) return true;
    ctx.setError(GL_INVALID_OPERATION);
    return false;
}

bool selecting(Context& ctx) { return ctx.state.fixed.renderMode == GL_SELECT; }

}  // namespace

GLint RenderMode(Context& ctx, GLenum mode) {
    if (!compatOnly(ctx)) return 0;
    if (mode != GL_RENDER && mode != GL_SELECT && mode != GL_FEEDBACK) {
        ctx.setError(GL_INVALID_ENUM);
        return 0;
    }
    if (mode != GL_RENDER)
        logOnce(LogLevel::Warn, LogCategory::Api, "glRenderMode",
                "glRenderMode(0x%04x): selection and feedback records are not produced", mode);
    ctx.state.fixed.renderMode = mode;
    ctx.state.fixed.nameStack.clear();
    return 0;
}

void SelectBuffer(Context& ctx, GLsizei size, GLuint* buffer) {
    (void)buffer;
    if (!compatOnly(ctx)) return;
    if (size < 0) return ctx.setError(GL_INVALID_VALUE);
    if (selecting(ctx)) ctx.setError(GL_INVALID_OPERATION);
}

void FeedbackBuffer(Context& ctx, GLsizei size, GLenum type, GLfloat* buffer) {
    (void)buffer;
    if (!compatOnly(ctx)) return;
    if (type != GL_2D && type != GL_3D && type != GL_3D_COLOR && type != GL_3D_COLOR_TEXTURE && type != GL_4D_COLOR_TEXTURE)
        return ctx.setError(GL_INVALID_ENUM);
    if (size < 0) return ctx.setError(GL_INVALID_VALUE);
    if (ctx.state.fixed.renderMode == GL_FEEDBACK) ctx.setError(GL_INVALID_OPERATION);
}

void PassThrough(Context& ctx, GLfloat token) {
    (void)token;
    compatOnly(ctx);
}

void InitNames(Context& ctx) {
    if (compatOnly(ctx) && selecting(ctx)) ctx.state.fixed.nameStack.clear();
}

void LoadName(Context& ctx, GLuint name) {
    if (!compatOnly(ctx) || !selecting(ctx)) return;
    std::vector<GLuint>& stack = ctx.state.fixed.nameStack;
    if (stack.empty()) return ctx.setError(GL_INVALID_OPERATION);
    stack.back() = name;
}

void PushName(Context& ctx, GLuint name) {
    if (!compatOnly(ctx) || !selecting(ctx)) return;
    std::vector<GLuint>& stack = ctx.state.fixed.nameStack;
    if (stack.size() >= kMaxNameStackDepth) return ctx.setError(GL_STACK_OVERFLOW);
    stack.push_back(name);
}

void PopName(Context& ctx) {
    if (!compatOnly(ctx) || !selecting(ctx)) return;
    std::vector<GLuint>& stack = ctx.state.fixed.nameStack;
    if (stack.empty()) return ctx.setError(GL_STACK_UNDERFLOW);
    stack.pop_back();
}

}  // namespace wmgl::api
