import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import hooks from 'eslint-plugin-react-hooks'
import globals from 'globals'

export default [
  { ignores: ['dist', 'node_modules'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  { files: ['playwright.config.ts', 'e2e/**/*.ts'], languageOptions: { globals: globals.node } },
  { files: ['src/**/*.{ts,tsx}'], languageOptions: { globals: globals.browser },
    plugins: { 'react-hooks': hooks },
    rules: { 'react-hooks/rules-of-hooks': 'error', 'react-hooks/exhaustive-deps': 'warn' } },
]
