# AutoKG Agent 静态工作台

## 在线演示

GitHub Pages: <https://ljttt0824.github.io/AutoKG-Agent/>

默认展示计算机网络课程图谱。可拖动节点、缩放和重排图谱，按实体类型筛选，搜索节点，查看节点/关系及其来源说明，并导出当前案例的 JSON。案例库中还保留通用知识图谱的脱敏演示样例。

公开页面由静态文件构成，使用 `public/demo-data/` 中的只读 fixture。构建时设置 `VITE_STATIC_DEMO=true` 后，页面不会请求本机 API 或模型服务，也不提供文件上传和新建任务入口。GitHub Pages 上的交互仅限本地浏览器中的案例查看。

## 本地运行

需要 Node.js 20 或更高版本。

```powershell
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173/`。开发服务器默认把 `/api` 请求代理到 `http://127.0.0.1:8000`；如果本地 FastAPI 后端已启动，可在本机使用真实文件解析和 DeepSeek 构建流程。模型密钥只配置在 `general-kg/.env`，不要放进 `web/.env`、前端变量或构建产物。

要本地预览与 GitHub Pages 相同的只读模式：

```powershell
$env:VITE_STATIC_DEMO = "true"
npm run build
npm run preview
```

## 课程案例来源与限制

- 计算机网络案例由课程项目生成的 `combined_graph.json` 中 Chapter 1 实体与关系精选而来，当前画布展示 18 个实体和 29 条关系；该项目完整输出记录为 2,840 个实体和 7,172 条关系。
- 本页重新撰写了实体解释和证据摘要，不包含课程课件、OCR 原文、原始数据库、个人信息或模型凭据。证据定位保留到课程章节和页码。
- 原构建结果没有实体/关系置信度字段，界面如实显示“未记录”。图谱关系属于自动抽取候选，尤其先修关系需要课程教师复核；Critic 分数未提供。
- 该静态页面是项目展示样例，不提供在线图谱构建服务，不接收用户上传，不适用于生产教学决策。
- 代码许可、课程资料许可和第三方组件许可彼此独立；课程材料没有随页面再发布。

## 发布

`.github/workflows/pages.yml` 将 `VITE_STATIC_DEMO=true` 的前端构建产物发布到 GitHub Pages。构建文件位于 `dist/`，不会上传本地后端、`.env`、课件或截图。
