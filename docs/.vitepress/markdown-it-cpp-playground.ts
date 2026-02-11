/**
 * markdown-it 插件：将 ```cpp-run 代码块自动转换为 <CppPlayground> 组件
 *
 * 用法：
 *   ```cpp-run title="可选标题"
 *   #include <iostream>
 *   int main() { ... }
 *   ```
 *
 * 原理：通过 encodeURIComponent 把代码编码后嵌入 Vue 模板属性，
 *       decodeURIComponent 是 Vue 模板允许的全局函数，运行时自动解码。
 */
import type MarkdownIt from 'markdown-it'

export function cppPlaygroundPlugin(md: MarkdownIt) {
  // 保存默认的 fence 渲染器
  const defaultFence =
    md.renderer.rules.fence ||
    function (tokens, idx, options, _env, self) {
      return self.renderToken(tokens, idx, options)
    }

  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const info = token.info.trim()

    // 匹配 ```cpp-run 或 ```cpp-run title="xxx"
    const match = info.match(/^cpp-run(?:\s+(.*))?$/)
    if (!match) {
      return defaultFence(tokens, idx, options, env, self)
    }

    const code = token.content
    const meta = match[1] || ''

    // 提取 title（可选）
    const titleMatch = meta.match(/title="([^"]*)"/)
    const title = titleMatch ? titleMatch[1] : ''

    // 用 encodeURIComponent 安全地编码代码内容
    // Vue 模板中 decodeURIComponent 是允许的全局函数
    const encoded = encodeURIComponent(code)

    const titleAttr = title ? ` title="${title}"` : ''

    return `<CppPlayground :code="decodeURIComponent('${encoded}')"${titleAttr} />\n`
  }
}
