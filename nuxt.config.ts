export default defineNuxtConfig({
  modules: [
    '@ant-design-vue/nuxt'
  ],
  devtools: { enabled: true },
  routeRules: {
    "/generate": { proxy: 'http://127.0.0.1:8000/generate' },
  },
})
