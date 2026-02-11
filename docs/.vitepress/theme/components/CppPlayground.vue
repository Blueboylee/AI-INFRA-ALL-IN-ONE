<script setup lang="ts">
import { ref, onMounted, watch, nextTick } from 'vue'

const props = defineProps<{
  code: string
  title?: string
  stdin?: string
}>()

const editableCode = ref('')
const output = ref<string | null>(null)
const running = ref(false)
const hasError = ref(false)
const textareaRef = ref<HTMLTextAreaElement | null>(null)

onMounted(() => {
  editableCode.value = props.code.trim()
  nextTick(autoResize)
})

watch(editableCode, () => {
  nextTick(autoResize)
})

function autoResize() {
  const el = textareaRef.value
  if (el) {
    el.style.height = 'auto'
    el.style.height = el.scrollHeight + 'px'
  }
}

function handleTab(e: KeyboardEvent) {
  const el = e.target as HTMLTextAreaElement
  const start = el.selectionStart
  const end = el.selectionEnd
  editableCode.value =
    editableCode.value.substring(0, start) +
    '    ' +
    editableCode.value.substring(end)
  nextTick(() => {
    el.selectionStart = el.selectionEnd = start + 4
  })
}

// 辅助函数：将 Godbolt 的 {text: string}[] 拼成字符串
function joinLines(lines: Array<{ text: string }> | undefined): string {
  if (!lines || lines.length === 0) return ''
  return lines.map((l) => l.text).join('\n')
}

async function runCode() {
  running.value = true
  output.value = null
  hasError.value = false

  try {
    // 使用 Godbolt (Compiler Explorer) API，CORS 支持良好
    const res = await fetch('https://godbolt.org/api/compiler/g132/compile', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        source: editableCode.value,
        options: {
          userArguments: '-std=c++20 -O2',
          executeParameters: {
            args: [],
            stdin: props.stdin || '',
          },
          compilerOptions: {
            executorRequest: true,
          },
          filters: {
            execute: true,
          },
        },
      }),
    })

    // 先读取原始文本，防止空响应导致 JSON.parse 崩溃
    const text = await res.text()

    if (!res.ok) {
      hasError.value = true
      output.value = `编译服务返回错误 (HTTP ${res.status}):\n${text || '(无响应体)'}`
      return
    }

    if (!text) {
      hasError.value = true
      output.value = '编译服务返回空响应，请稍后重试。'
      return
    }

    const data = JSON.parse(text)
    let result = ''

    // 1. 编译阶段的错误/警告
    const buildStderr = joinLines(data.buildResult?.stderr)
    if (buildStderr) {
      result += buildStderr
    }

    // 2. 编译失败（code != 0）
    if (data.buildResult?.code !== 0) {
      hasError.value = true
      output.value = result || `编译失败 (exit code: ${data.buildResult?.code})`
      return
    }

    // 3. 运行阶段输出
    const stdout = joinLines(data.stdout)
    const stderr = joinLines(data.stderr)

    if (stdout) {
      if (result) result += '\n'
      result += stdout
    }
    if (stderr) {
      if (result) result += '\n'
      result += stderr
    }

    // 4. 运行阶段异常退出
    if (data.code !== 0) {
      hasError.value = true
      if (!result) result = `程序退出码: ${data.code}`
    }

    output.value = result || '(无输出)'
  } catch (e: any) {
    output.value = `运行失败: ${e.message || e}`
    hasError.value = true
  } finally {
    running.value = false
  }
}

function resetCode() {
  editableCode.value = props.code.trim()
  output.value = null
  hasError.value = false
}
</script>

<template>
  <div class="cpp-playground">
    <div class="playground-header">
      <div class="header-left">
        <span class="lang-badge">C++</span>
        <span v-if="title" class="playground-title">{{ title }}</span>
      </div>
      <div class="header-right">
        <button class="btn btn-reset" @click="resetCode" title="重置代码">
          ↺ 重置
        </button>
        <button
          class="btn btn-run"
          @click="runCode"
          :disabled="running"
          :class="{ 'is-running': running }"
        >
          <span v-if="running" class="spinner" />
          {{ running ? '编译运行中...' : '▶ 运行' }}
        </button>
      </div>
    </div>

    <div class="editor-wrapper">
      <textarea
        ref="textareaRef"
        v-model="editableCode"
        spellcheck="false"
        autocomplete="off"
        autocorrect="off"
        autocapitalize="off"
        @keydown.tab.prevent="handleTab"
      />
    </div>

    <div v-if="output !== null" class="output-section" :class="{ 'has-error': hasError }">
      <div class="output-header">
        {{ hasError ? '⚠ 错误' : '✓ 输出' }}
      </div>
      <pre class="output-content">{{ output }}</pre>
    </div>
  </div>
</template>

<style scoped>
.cpp-playground {
  margin: 16px 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
  background: var(--vp-c-bg-soft);
}

.playground-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: var(--vp-c-bg-alt);
  border-bottom: 1px solid var(--vp-c-divider);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.lang-badge {
  display: inline-block;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 600;
  color: var(--vp-c-white);
  background: #00599c;
  border-radius: 4px;
}

.playground-title {
  font-size: 13px;
  color: var(--vp-c-text-2);
  font-weight: 500;
}

.header-right {
  display: flex;
  gap: 6px;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 12px;
  font-size: 13px;
  font-weight: 500;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-reset {
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
}

.btn-reset:hover {
  color: var(--vp-c-text-1);
  border-color: var(--vp-c-text-3);
}

.btn-run {
  background: var(--vp-c-brand-1);
  color: var(--vp-c-white);
  border-color: var(--vp-c-brand-1);
}

.btn-run:hover:not(:disabled) {
  background: var(--vp-c-brand-2);
  border-color: var(--vp-c-brand-2);
}

.btn-run:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.btn-run.is-running {
  background: var(--vp-c-brand-3);
  border-color: var(--vp-c-brand-3);
}

.spinner {
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.editor-wrapper {
  position: relative;
}

.editor-wrapper textarea {
  display: block;
  width: 100%;
  min-height: 120px;
  padding: 16px;
  margin: 0;
  font-family: var(--vp-font-family-mono);
  font-size: 14px;
  line-height: 1.6;
  color: var(--vp-c-text-1);
  background: var(--vp-code-block-bg);
  border: none;
  outline: none;
  resize: none;
  overflow: hidden;
  tab-size: 4;
  -moz-tab-size: 4;
  box-sizing: border-box;
}

.editor-wrapper textarea:focus {
  box-shadow: inset 0 0 0 1px var(--vp-c-brand-1);
}

.output-section {
  border-top: 1px solid var(--vp-c-divider);
}

.output-header {
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 600;
  color: var(--vp-c-green-2);
  background: var(--vp-c-bg-alt);
  border-bottom: 1px solid var(--vp-c-divider);
}

.has-error .output-header {
  color: var(--vp-c-red-2);
}

.output-content {
  margin: 0;
  padding: 12px 16px;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  line-height: 1.5;
  color: var(--vp-c-text-1);
  background: var(--vp-code-block-bg);
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
}

.has-error .output-content {
  color: var(--vp-c-red-1);
}
</style>
