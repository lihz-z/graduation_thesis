# 毕业论文项目说明

本仓库是我的本科毕业论文项目，论文题目为《面向高级驾驶辅助系统的雨天条件下车道线鲁棒检测方法研究》。

项目 GitHub 地址：

- 仓库主页：https://github.com/lihz-z/graduation_thesis
- 在线查看 `main.pdf`：https://github.com/lihz-z/graduation_thesis/blob/main/main.pdf
- 直接下载 `main.pdf`：https://github.com/lihz-z/graduation_thesis/blob/main/main.pdf?raw=true

## 模板说明

本项目基于 `sysuthesis` LaTeX 模板进行撰写，并在此基础上结合我自己的论文内容进行了修改与扩展。

模板仓库链接：

- https://github.com/1FCENdoge/sysuthesis

感谢原模板作者和维护者提供的工作基础。

## 项目内容

本仓库保存了论文写作、图片资源、参考文献和最终生成结果等内容，主要包括：

- `main.tex`：论文主入口文件
- `sysusetup.tex`：论文题目、作者、院系、导师等基础信息配置
- `docs/`：各章节正文内容
- `image/`：论文中使用的图片资源
- `reference.bib`：参考文献数据库
- `main.pdf`：已经编译好的论文成品

如果你只是想阅读论文内容，直接下载 `main.pdf` 即可，不需要本地编译。

## 如何使用

### 1. 直接查看或下载论文

本项目已经附带编译完成的 `main.pdf`，可直接通过下面链接下载：

- https://github.com/lihz-z/graduation_thesis/blob/main/main.pdf?raw=true

### 2. 本地编译论文

建议使用支持 `XeLaTeX` 的 TeX Live 环境进行编译。

在项目根目录执行：

```bash
make main
```

或者直接执行：

```bash
latexmk -xelatex main.tex
```

编译完成后会在根目录生成新的 `main.pdf`。

### 3. 修改论文内容

常见修改位置如下：

- 修改论文题目、作者、导师、学号等信息：编辑 `sysusetup.tex`
- 修改正文内容：编辑 `docs/` 下对应章节文件
- 修改图片：将图片放入 `image/` 对应目录，并在正文中更新路径
- 修改参考文献：编辑 `reference.bib`

## 使用建议

- 如果只是提交或分享论文结果，优先使用仓库中的 `main.pdf`
- 如果需要继续修改论文内容，建议只编辑源码文件后重新编译
- 编译过程中产生的中间文件通常不需要再次提交

## 说明

本仓库主要用于保存和展示我的毕业论文项目，不作为通用模板发布。若你需要撰写自己的中山大学 LaTeX 论文，建议从上方给出的模板仓库开始使用。
