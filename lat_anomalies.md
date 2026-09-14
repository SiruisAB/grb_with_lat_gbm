# LAT 下载异常记录

记录 GBM+LAT 联合目标中，LAT 数据无法按标准流程下载或需要特殊处理的暴。

---

## GRB 260411B (bn260411337)

**类型**：LLE-only 探测，标准 >100 MeV 分析无显著超出

**GCN 来源**：#44262 (Fermi GBM Final Real-time Localization)、#44277 (Fermi-LAT detection)

**关键事实**（引自 #44277）：

> The GBM location was initially inside the LAT field of view at an angle of 60 degrees
> to the LAT boresight, and remained in the LAT field of view until ~T0+500 s.
> No significant excess is seen using standard >100 MeV likelihood analysis procedures.
>
> Using the LAT Low Energy (LLE) data selection, over 190 counts above background were
> detected within a 20 s interval coinciding with the time of the GBM emission. This data
> selection has insufficient spatial resolution to provide a reliable LAT localization.
> Since an excess of events was not seen using the standard analysis selection, this
> detection is likely due to low-energy gamma-rays (below 100 MeV).

**导致的问题**：

1. LAT circular 未给出 LAT 定位（LLE 空间分辨率不足）→ `lat_ra` / `lat_dec` 为空
2. LAT circular 未给出标准分析时窗 → `lat_t0` / `lat_t1` 为空
3. 因此 `gcn_fermi_bulk.json` 中该暴全部 `lat_*` 字段为空字符串，自动化流程
   按"字段不完整"跳过。这不是解析 bug，是数据本身没有。

**采取的处理**：

改用 GBM 定位 + LAT 视场停留时窗下载 Extended 数据。

| 参数 | 值 | 来源 |
|---|---|---|
| RA, Dec | 83.2, 2.8 (J2000) | GCN #44262，统计误差 1.0° |
| trigger_met | 797587480 | GBM trigger 797587480.306254 |
| t0, t1 | 0, 500 s | LAT 视场停留时间（#44277: "until ~T0+500 s"） |
| LAT boresight | 60.0° | #44262 / #44277 |
| MET 时窗 | 797586480 – 797589980 | t0-1000 → t1+2000（脚本默认 pad） |
| 能段 | 100 MeV – 100 GeV | 脚本默认 |
| 半径 | 12° | 脚本默认 |

**使用这份数据时必须知道**：

- 下载的是标准 100 MeV–100 GeV Extended 数据，而该暴信号在 100 MeV **以下**，
  所以这份数据大概率是纯背景，不应期待标准似然分析出显著结果。
- 真正的信号在 LLE 数据里。LLE 是独立数据产品，不在 Extended 查询服务范围内，
  需从 FSSC LLE 目录另行获取。
- 定位用的是 GBM 坐标，误差 1.0°，远大于常规 LAT 定位的 ~0.1°。

**GBM 侧可用参数**（#44266）：

- T0: 08:04:35.31 UT on 11 April 2026
- fluence: (2.31 ± 0.05)E-05 erg/cm^2 [10-1000 keV]
- photon flux: (21.8 ± 0.4) ph/cm2/s
- Epeak: 348 ± 4 keV
- photon index: -0.62 ± 0.02
- 谱型: power law with exponential high-energy cutoff
- GBM 谱积分时窗: T0-2.9 s 到 T0+29.2 s

---

## 2026 年样本补入目录表（2026-08-11）

**背景**：12 个 2026 年暴的 GBM 与 LAT 数据都已落盘，但 web 端下拉选不到。
原因在 `web_app.py:1269` —— 候选项是 `GBMcatolog.xls` 与 `fermilat-grb.xls`
按 bnname 求**交集**得到的，与磁盘上有没有数据无关。补入前 LAT 表 2026 条目为 0，
交集为空。

**已补入的 12 个暴**：

| bnname | GRB | trigger_met | RA, Dec | T0–T1 |
|---|---|---|---|---|
| bn260208214 | GRB260208A | 792220053 | 204.57, 33.77 | 400–2500 |
| bn260208412 | GRB260208B | 792237169 | 99.64, -13.84 | 0–2000 |
| bn260226443 | GRB260226A | 793795080.95811 | 41.93, 7.73 | 0–1500 |
| bn260305112 | GRB260305A | 794371303.18619 | 218.73, 35.49 | 0–100 |
| bn260410294 | GRB260410A | 797497442 | 110.67, -20.94 | 0–2 |
| bn260411337 | GRB260411B | 797587480.306254 | 83.2, 2.8 | 0–500 |
| bn260522094 | GRB260522A | 801108913 | 346.67, 9.66 | 0–50 |
| bn260607555 | GRB260607B | 802531173 | 54.68, 30.45 | 0–1200 |
| bn260616140 | GRB260616A | 803272919 | 93.53, -32.07 | 500–3000 |
| bn260618705 | GRB260618B | 803494509 | 354.35, -44.88 | 0–100 |
| bn260708424 | GRB260708A | 805198281 | 225.61, -44.07 | 0–400 |
| bn260714766 | GRB260714B | 805746211.760106 | 163.12, 46.98 | 0–400 |

**数据口径**：

- LAT 表（GCN sheet）：ra/dec/T0/T1 取自各暴 LAT circular，并与已下载 Extended
  数据的 FITS 头 `DSVAL(POS)`、`TSTART/TSTOP` 逐条核对一致。
- GBM 表（fermigbrst sheet）：t90/t90_start/fluence/trigger_time 取自各暴 bcat
  文件头（GBM catalog 各列的原始来源），不是手抄 circular 数值。
- 背景窗列留空，`_background_defaults` 会回退到 `DEFAULT_BACKGROUND_LOW/HIGH`
  (-24--5 / 350-400)。需要逐暴调的在 web 端改或写进 `special_bursts.yaml`。

**两条需要注意的取值**：

1. **GRB260226A** 用的是 refined analysis (#43850) 的 RA,Dec = 41.93, 7.73
   （误差 0.11°），不是首报 #43844 的 42.050, +8.033（误差 0.5°）。
   `GRB_FermiLAT_Time_Extended.csv` 里这条是 N/A，未采用。
2. **GRB260714B** 没有 LAT detection circular，只有 GBM (#45158) 与
   BALROG (#45160)。表中坐标 163.12, 46.98 取自已下载数据的 FITS 头
   （与 BALROG 的 163.3, 47.8 接近，非 GBM 首报的 166.4, 46.7，误差 4.0°）。
   **该暴的 LAT 探测未经 LAT 团队确认，做联合分析时需自行判断显著性。**

**GRB260411B 仍按上文特殊性对待**：表中 T0–T1 填的是 LAT 视场停留时窗 0–500 s、
坐标是 GBM 定位，但其 >100 MeV 数据实为纯背景（已验证：3500 s 内 12° 半径仅 64 个
事件，无时间聚集）。它在 web 端会显示为普通 gbm+lat 样本，**不要直接采信其 LAT
联合拟合结果**。

**写入方式**（两表格式不同，均已备份为 `*.bak_pre2026`）：

- `fermilat-grb.xls` 扩展名是 .xls 但实为 xlsx，openpyxl 按扩展名拒绝，
  经 `io.BytesIO` 读写绕开。
- `GBMcatolog.xls` 是真 BIFF .xls，pandas 2.x 已移除 xlwt 写入引擎，
  改用 xlrd 读 + xlwt 全量重写（xlwt 已在 fermipy 环境 pip 安装）。
- 回归验证：两表原有行逐格未变，LAT 工作簿 6 个 sheet 行列数除 GCN 外均未变。
