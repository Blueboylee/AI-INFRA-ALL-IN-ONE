import DefaultTheme from 'vitepress/theme'
import CppPlayground from './components/CppPlayground.vue'
import type { Theme } from 'vitepress'

const theme: Theme = {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('CppPlayground', CppPlayground)
  },
}

export default theme
