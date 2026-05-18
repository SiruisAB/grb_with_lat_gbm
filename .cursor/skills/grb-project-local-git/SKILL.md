---
name: grb-project-local-git
description: >-
  Commits changes in the grb_project Git repo after code or doc edits. Use when
  modifying files under /home/mxr/lee/gbmtest/grb_project, when the user asks to
  sync or update the local repository, or at the end of any task that changed
  grb_project (Python modules, 说明文档.md, Cursor config in this repo).
---

# grb_project 本地 Git 同步

## 仓库信息

| 项 | 值 |
|----|-----|
| 根目录 | `/home/mxr/lee/gbmtest/grb_project` |
| 默认分支 | `main` |
| 远程 | 仅当用户明确要求时再 `git push` |

## 何时执行

在以下情况**任务结束前**必须执行本 skill 的提交流程：

- 修改了 `grb_project` 内任意源码、配置或文档
- 用户要求「更新本地仓库」「提交一下」「同步 git」
- 完成了一项针对本项目的实现、修复或重构

## 提交流程

1. 进入仓库根目录并查看状态：
   ```bash
   cd /home/mxr/lee/gbmtest/grb_project
   git status
   git diff
   git diff --staged   # 若有已暂存文件
   git log -5 --oneline   # 对齐既有 commit 风格
   ```

2. **只暂存**与本次任务相关的路径：
   ```bash
   git add <paths>
   ```
   不要 `git add .` 盲目全加。

3. **禁止提交**（除非用户明确要求）：
   - `.env`、密钥、token、凭据文件
   - 大体积结果目录（如 `results*`、原始拟合输出）
   - 无关临时文件、编辑器垃圾

4. 用 HEREDOC 提交，message 1–2 句，强调 **why**：
   ```bash
   git commit -m "$(cat <<'EOF'
   <type>(<scope>): <简短说明>

   <可选：一句补充原因>
   EOF
   )"
   ```

5. 再次 `git status`，确认工作区干净或仅剩与本次任务无关的未跟踪文件。

6. 在回复中简要列出：commit hash、message、涉及文件。无改动则说明「无需提交」。

## Commit 风格（与历史一致）

常见 `type`：`feat`、`fix`、`refactor`、`docs`、`chore`。

示例（来自本仓库）：

```
refactor(export): 移除最佳模型 CSV/TeX 导出
feat(fit): 增加 bb+pl/mbb 模型并调整嵌套采样 live 点数
docs: 同步说明文档与当前导出/流程行为
```

逻辑上独立的改动应**分多次 commit**，不要混在一个 message 里。

## 安全约束

- 不要 `git config` 修改
- 不要 `push --force`、`reset --hard` 等破坏性命令，除非用户明确要求
- 不要 `--no-verify` / `--no-gpg-sign`，除非用户明确要求
- 不要 `git commit --amend`，除非用户明确要求且满足 amend 安全条件

## 多 commit 检查清单

```
- [ ] git status / diff 已查看
- [ ] 仅添加任务相关文件
- [ ] message 说明 why，风格与 git log 一致
- [ ] git commit 成功
- [ ] git status 已确认
- [ ] 未擅自 push
```
