import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import re
import io
import os
import tempfile

# FeRAM P-E Hysteresis Analyzer 
# ============================================================
# 此網頁是為了方便直觀顯示FeRAM的相關數據圖，意圖整合excel及Origin的功能
# V19(每跑不動or功能有問題就記錄一次)
# 特別感謝王鑫-{輕鬆又漂亮的Python Web框架} & Gemini pro & Claude pro & Github大老，小弟受寵若驚
# 電物116謝文豪敬上
# ============================================================
st.set_page_config(page_title="FeRAM Lab", layout="wide")
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 11})


# ============================================================
# A. 檔案讀取
# ============================================================
def load_file(file_obj, filename):
    file_obj.seek(0)
    raw_bytes = file_obj.read()
    file_obj.seek(0)
    try:
        text = raw_bytes.decode('utf-8', errors='replace')
        if 'DataName,' in text and 'DataValue,' in text:
            lines = text.split('\n')
            header_line = data_start = None
            for i, line in enumerate(lines):
                if line.strip().startswith('DataName,'):
                    header_line = i
                elif line.strip().startswith('DataValue,') and data_start is None:
                    data_start = i
            if header_line is not None and data_start is not None:
                headers = [h.strip() for h in lines[header_line].split(',')[1:]]
                rows = []
                for line in lines[data_start:]:
                    if line.strip().startswith('DataValue,'):
                        vals = line.split(',')[1:]
                        try:
                            rows.append([float(v.strip()) for v in vals if v.strip()])
                        except ValueError:
                            continue
                if rows:
                    return pd.DataFrame(rows, columns=headers[:len(rows[0])])
    except Exception:
        pass

    file_obj.seek(0)
    try:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            preview = pd.read_excel(file_obj, header=None, nrows=50)
            h_idx = 0
            for i, row in preview.iterrows():
                s = str(row.values).lower()
                if 'measresult1' in s or 'voltage' in s or ('time' in s and 'meas' in s):
                    h_idx = i; break
            file_obj.seek(0)
            return pd.read_excel(file_obj, header=h_idx)
        else:
            text = raw_bytes.decode('utf-8', errors='ignore')
            lines = text.split('\n')
            h_idx = 0
            for i, line in enumerate(lines[:50]):
                s = line.lower()
                if 'measresult1' in s or 'voltage' in s or ('time' in s and 'meas' in s):
                    h_idx = i; break
            return pd.read_csv(io.StringIO(text), skiprows=h_idx)
    except Exception as e:
        st.error(f"檔案解析錯誤 ({filename}): {e}")
        return None


def extract_archive(uploaded_file):
    name = uploaded_file.name
    files = []
    if name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            for f in z.namelist():
                if f.endswith(('.xlsx', '.csv')) and '__MACOSX' not in f and '~$' not in f:
                    files.append((os.path.basename(f), io.BytesIO(z.read(f))))
    elif name.endswith('.7z'):
        try:
            import py7zr
        except ImportError:
            st.error("缺少 py7zr：`pip install py7zr`")
            return []
        try:
            with py7zr.SevenZipFile(uploaded_file, mode='r') as z:
                for fname, bio in z.readall().items():
                    if fname.endswith(('.xlsx', '.csv')) and '__MACOSX' not in fname and '~$' not in fname:
                        bio.seek(0)
                        files.append((os.path.basename(fname), io.BytesIO(bio.read())))
        except Exception as e:
            st.error(f".7z 解壓縮失敗: {e}")
    else:
        files.append((name, uploaded_file))
    return files


# ============================================================
# B. PUND 分析器
# ============================================================
class FerroelectricAnalyzer:
    def __init__(self, df, area_um2, filename="Unknown"):
        self.df = df
        self.filename = filename
        self.area_um2 = area_um2
        self.area_cm2 = area_um2 * 1e-8
        self.error_msg = None
        try:
            self._map_columns()
            self._parse_cycle_count()
        except Exception as e:
            self.error_msg = f"Init: {e}"

    def _map_columns(self):
        cols_lower = [str(c).lower().strip() for c in self.df.columns]
        v_keys = ["measresult1_value", "measresult1", "voltage", "force", "voltage_1"]
        i_keys = ["measresult2_value", "measresult2", "current", "current_1"]
        t_keys = ["time", "measresult1_time"]
        def find_col(keys):
            for i, c in enumerate(cols_lower):
                if any(k in c for k in keys): return i
            return None
        v_idx, i_idx, t_idx = find_col(v_keys), find_col(i_keys), find_col(t_keys)
        if v_idx is not None and i_idx is not None:
            self.V = pd.to_numeric(self.df.iloc[:, v_idx], errors='coerce').values
            self.I = pd.to_numeric(self.df.iloc[:, i_idx], errors='coerce').values
            self.T = (pd.to_numeric(self.df.iloc[:, t_idx], errors='coerce').values
                      if t_idx is not None else np.arange(len(self.V)) * 5e-7)
        elif self.df.shape[1] >= 3:
            self.T = pd.to_numeric(self.df.iloc[:, 0], errors='coerce').values
            self.V = pd.to_numeric(self.df.iloc[:, 1], errors='coerce').values
            self.I = pd.to_numeric(self.df.iloc[:, 2], errors='coerce').values
        else:
            raise ValueError("無法識別 V/I/T 欄位")
        mask = ~np.isnan(self.V) & ~np.isnan(self.I) & ~np.isnan(self.T)
        self.V, self.I, self.T = self.V[mask], self.I[mask], self.T[mask]
        if len(self.T) > 20:
            resets = np.where(np.diff(self.T) <= 0)[0]
            if len(resets) > 0:
                self.V = self.V[:resets[0]+1]
                self.I = self.I[:resets[0]+1]
                self.T = self.T[:resets[0]+1]

    def _parse_cycle_count(self):
        match = re.search(r'1E(\d+)', self.filename, re.IGNORECASE)
        if match:
            self.cycle_pow = int(match.group(1))
            self.cycle_num = 10 ** self.cycle_pow
            self.cycle_label = f"1E{self.cycle_pow}"
        else:
            self.cycle_num = 1; self.cycle_pow = 0; self.cycle_label = "Fresh"

    def _detect_pulses(self, v_th):
        n = len(self.V)
        active = np.abs(self.V) > v_th
        diff = np.diff(active.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]
        if active[0]: starts = np.concatenate([[0], starts])
        if active[-1]: ends = np.concatenate([ends, [n - 1]])
        if len(starts) == 0 or len(ends) == 0: return []
        if starts[0] > ends[0]: ends = ends[1:]
        if len(starts) > len(ends): starts = starts[:len(ends)]
        merged_s, merged_e = [starts[0]], []
        for i in range(1, len(starts)):
            if starts[i] - ends[i-1] < 3: pass
            else: merged_e.append(ends[i-1]); merged_s.append(starts[i])
        merged_e.append(ends[-1])
        pulses = []
        for s, e in zip(merged_s, merged_e):
            if e - s < 5: continue
            mid = (s + e) // 2
            pulses.append({'s': s, 'e': e, 'pol': 1 if self.V[mid] > 0 else -1})
        return pulses

    def _assign_pund(self, pulses):
        if len(pulses) < 4: return None, None
        target = pulses[1:] if len(pulses) >= 5 else pulses
        pair1 = pair2 = None
        for i in range(len(target) - 1):
            if target[i]['pol'] == target[i+1]['pol']:
                if pair1 is None: pair1 = (target[i], target[i+1])
                elif pair2 is None: pair2 = (target[i], target[i+1]); break
        return pair1, pair2

    def _phase_lock(self, pulse_sw, pulse_nsw):
        s1, e1 = pulse_sw['s'], pulse_sw['e']
        s2 = pulse_nsw['s']
        n = len(self.V); pw = e1 - s1
        period_est = s2 - s1; best_shift = period_est; min_err = float('inf')
        search = min(50, max(pw // 4, 15))
        for shift in range(max(1, period_est - search), min(n - s1 - pw, period_est + search + 1)):
            if s1 + shift + pw > n: continue
            err = np.sum(np.abs(self.V[s1:s1+pw] - self.V[s1+shift:s1+shift+pw]))
            if err < min_err: min_err = err; best_shift = shift
        max_len = min(pw, n - (s1 + best_shift))
        if max_len <= 0: return None, None, None, 0
        return (self.I[s1:s1+max_len], self.I[s1+best_shift:s1+best_shift+max_len],
                self.V[s1:s1+max_len], max_len)

    def calculate(self, v_th=0.01, invert=False, linear_comp=False, v_invert=False, **kw):
        if self.error_msg: return None, self.error_msg
        try:
            return self._do_calculate(v_th, invert, linear_comp, v_invert)
        except Exception as e:
            return None, f"計算錯誤: {e}"

    def _do_calculate(self, v_th, invert, linear_comp, v_invert):
        if len(self.T) > 1:
            diffs = np.diff(self.T)
            dt = np.median(diffs[diffs > 0])
            if dt > 0.01: dt *= 1e-6
        else:
            dt = 5e-7

        pulses = self._detect_pulses(v_th)
        if len(pulses) < 4:
            return None, f"脈衝數不足({len(pulses)})"

        pair1, pair2 = self._assign_pund(pulses)
        if pair1 is None or pair2 is None:
            return None, "找不到配對"

        I_sw1, I_nsw1, V_1, len1 = self._phase_lock(pair1[0], pair1[1])
        I_sw2, I_nsw2, V_2, len2 = self._phase_lock(pair2[0], pair2[1])
        if I_sw1 is None or I_sw2 is None or len1 < 5 or len2 < 5:
            return None, "相位鎖定失敗"

        I_ferro_1 = I_sw1 - I_nsw1
        I_ferro_2 = I_sw2 - I_nsw2

        V_1_out = -V_1 if v_invert else V_1
        V_2_out = -V_2 if v_invert else V_2
        if invert:
            I_ferro_1 = -I_ferro_1
            I_ferro_2 = -I_ferro_2

        I_ferro = np.concatenate([I_ferro_1, I_ferro_2])
        V_loop = np.concatenate([V_1_out, V_2_out])

        Q = np.cumsum(I_ferro * dt)
        P = (Q / self.area_cm2) * 1e6

        if linear_comp and len(P) > 1:
            delta = P[-1] - P[0]
            P -= np.linspace(0, delta, len(P))

        shift = (np.max(P) + np.min(P)) / 2
        P -= shift

        n_pos = len(I_ferro_1)
        Two_Pr, Two_Vc = self._extract_params(V_loop, P, n_pos)

        # ── 進階特徵（v20 新增）──
        adv = self._extract_advanced(V_loop, P, n_pos, I_ferro_1, I_ferro_2,
                                      I_sw1, I_nsw1, I_sw2, I_nsw2,
                                      V_1, V_2, dt)

        return {
            "cycle_num": self.cycle_num,
            "cycle_label": self.cycle_label,
            "filename": self.filename,
            "2Pr": Two_Pr,
            "2Vc": Two_Vc,
            "data_P": P,
            "data_V": V_loop,
            "n_pos": n_pos,
            "n_pulses": len(pulses),
            "dt": dt,
            # v20: 存中間數據供 debug 視覺化
            "I_ferro_1": I_ferro_1,
            "I_ferro_2": I_ferro_2,
            "I_nsw1": I_nsw1,
            "I_nsw2": I_nsw2,
            "V_1": V_1,
            "V_2": V_2,
            "advanced": adv,
        }, None

    def _extract_params(self, V, P, n_pos):
        def get_pr(v_seg, p_seg):
            if len(v_seg) < 2: return 0
            pk = np.argmax(np.abs(v_seg))
            rv, rp = v_seg[pk:], p_seg[pk:]
            return rp[np.argmin(np.abs(rv))] if len(rv) > 0 else 0
        Pr_pos = get_pr(V[:n_pos], P[:n_pos])
        Pr_neg = get_pr(V[n_pos:], P[n_pos:])
        Two_Pr = abs(Pr_pos) + abs(Pr_neg)
        Two_Vc = 0.0
        zc = np.where(np.diff(np.sign(P)))[0]
        if len(zc) >= 2:
            Two_Vc = np.max(V[zc]) - np.min(V[zc])
        return Two_Pr, Two_Vc

    # ── v20 新增：進階物理特徵提取 ──
    def _extract_advanced(self, V_loop, P, n_pos,
                          I_ferro_1, I_ferro_2,
                          I_sw1, I_nsw1, I_sw2, I_nsw2,
                          V_1, V_2, dt):
        """計算 12 項分析所需的所有物理量"""
        adv = {}

        # ── 1. Pr+, Pr-, 2Pr (已在 _extract_params 算過，這裡再算一次存到 adv) ──
        def get_pr(v, p):
            pk = np.argmax(np.abs(v))
            rv, rp = v[pk:], p[pk:]
            return rp[np.argmin(np.abs(rv))] if len(rv) > 0 else 0

        adv['Pr_pos'] = get_pr(V_loop[:n_pos], P[:n_pos])
        adv['Pr_neg'] = get_pr(V_loop[n_pos:], P[n_pos:])
        adv['2Pr'] = abs(adv['Pr_pos']) + abs(adv['Pr_neg'])

        # ── 2. Vc+, Vc-, 2Vc ──
        zc = np.where(np.diff(np.sign(P)))[0]
        if len(zc) >= 2:
            adv['Vc_pos'] = np.max(V_loop[zc])
            adv['Vc_neg'] = np.min(V_loop[zc])
        else:
            adv['Vc_pos'] = 0
            adv['Vc_neg'] = 0
        adv['2Vc'] = abs(adv['Vc_pos']) + abs(adv['Vc_neg'])

        # ── 3. Imprint (Vc 偏移) ──
        adv['imprint'] = (adv['Vc_pos'] + adv['Vc_neg']) / 2

        # ── 4. P_sat, Squareness ──
        adv['P_sat'] = abs(P[np.argmax(np.abs(V_loop[:n_pos]))])
        adv['squareness'] = abs(adv['Pr_pos']) / (adv['P_sat'] + 1e-10)

        # ── 5. Pr 不對稱性 ──
        adv['Pr_asym'] = ((abs(adv['Pr_pos']) - abs(adv['Pr_neg'])) /
                          (abs(adv['Pr_pos']) + abs(adv['Pr_neg']) + 1e-10))

        # ── 6. Loop area（能量損耗）──
        try:
            adv['loop_area'] = abs(np.trapezoid(P, V_loop))
        except AttributeError:
            adv['loop_area'] = abs(np.trapz(P, V_loop))

        # ── 7. Switching current 分析（用正半迴線 I_ferro_1）──
        I_abs = np.abs(I_ferro_1)
        adv['I_sw_peak'] = np.max(I_abs) * 1e6  # µA
        peak_idx = np.argmax(I_abs)
        adv['t_peak'] = peak_idx * dt * 1e6  # µs

        # FWHM
        half_max = adv['I_sw_peak'] * 1e-6 / 2
        above_half = I_abs > half_max
        fwhm_pts = np.sum(above_half)
        adv['FWHM'] = fwhm_pts * dt * 1e6  # µs

        # t_90: 90% 電荷切換完成的時間
        Q_ferro = np.cumsum(np.abs(I_ferro_1) * dt)
        if Q_ferro[-1] > 0:
            t90_idx = np.searchsorted(Q_ferro, Q_ferro[-1] * 0.9)
            adv['t_90'] = t90_idx * dt * 1e6
        else:
            adv['t_90'] = 0

        # dI/dt max
        if len(I_ferro_1) > 2:
            dI = np.diff(I_ferro_1) / dt
            adv['dI_dt_max'] = np.max(np.abs(dI)) * 1e-6  # A/s → µA/µs
        else:
            adv['dI_dt_max'] = 0

        # ── 8. Leakage current（falling branch + U pulse tail）──
        pk1 = np.argmax(np.abs(V_1))
        if pk1 < len(I_ferro_1) - 5:
            adv['I_leak'] = np.mean(np.abs(I_ferro_1[pk1:])) * 1e6
        else:
            adv['I_leak'] = 0

        # I_tail: U 脈衝（non-switching）的後 20%
        tail_start = int(len(I_nsw1) * 0.8)
        if tail_start < len(I_nsw1):
            adv['I_tail'] = np.mean(np.abs(I_nsw1[tail_start:])) * 1e6
        else:
            adv['I_tail'] = 0

        # ── 9. NLS model 擬合 ──
        try:
            from scipy.optimize import curve_fit
            Q_sw = np.cumsum(np.abs(I_ferro_1) * dt)
            if Q_sw[-1] > 0:
                Q_norm = Q_sw / Q_sw[-1]
                t_arr = np.arange(len(Q_norm)) * dt
                t_arr = t_arr + 1e-10  # avoid log(0)

                def kai(t, t0, n):
                    return 1 - np.exp(-(t / t0) ** n)

                popt, _ = curve_fit(kai, t_arr, Q_norm,
                                    p0=[t_arr[len(t_arr)//2], 2.0],
                                    bounds=([1e-9, 0.1], [1.0, 10.0]),
                                    maxfev=5000)
                adv['NLS_t0'] = popt[0] * 1e6  # µs
                adv['NLS_n'] = popt[1]

                Q_fit = kai(t_arr, *popt)
                ss_res = np.sum((Q_norm - Q_fit) ** 2)
                ss_tot = np.sum((Q_norm - Q_norm.mean()) ** 2)
                adv['NLS_R2'] = 1 - ss_res / (ss_tot + 1e-20)
            else:
                adv['NLS_t0'] = 0; adv['NLS_n'] = 0; adv['NLS_R2'] = 0
        except Exception:
            adv['NLS_t0'] = 0; adv['NLS_n'] = 0; adv['NLS_R2'] = 0

        return adv


# ============================================================
# C. 自動元件分類
# ============================================================
def auto_classify_device(results):
    """根據所有 cycle 的物理特徵自動分類"""
    if len(results) < 2:
        return 'unknown', '數據不足'

    advs = [r.get('advanced', {}) for r in results]
    prs = [a.get('2Pr', 0) for a in advs]
    areas = [a.get('loop_area', 0) for a in advs]
    sqs = [a.get('squareness', 0) for a in advs]
    leaks = [a.get('I_leak', 0) for a in advs]

    # Dead: 極端值
    if prs[0] > 40 or prs[0] < 0.5:
        return 'dead', f'初始 2Pr={prs[0]:.1f} 異常'

    # Breakdown: 2Pr 驟降 >40%
    pr_max = max(prs)
    for i in range(1, len(prs)):
        if prs[i] < pr_max * 0.6:
            return 'breakdown', f'2Pr 從 {pr_max:.1f} 降至 {prs[i]:.1f} at cycle {i}'
        if i >= 2 and areas[i] > areas[0] * 2.0:
            return 'breakdown', f'Loop area 暴增 {areas[i]/areas[0]:.1f}× at cycle {i}'

    # Leaky
    if max(sqs) < 0.5:
        avg_leak_ratio = np.mean(leaks) / (np.mean([a.get('I_sw_peak', 1) for a in advs]) + 1e-10)
        if avg_leak_ratio > 0.3:
            return 'leaky', f'Sq<0.5, leak/switch ratio={avg_leak_ratio:.2f}'

    if max(prs) < 5.0 and np.std(prs) / (np.mean(prs) + 1e-10) > 0.3:
        return 'leaky', f'2Pr<5 且變異大'

    # Good（包括 wake-up）
    wake_type = 'stable'
    if len(prs) >= 3 and prs[-1] > prs[0] * 1.3:
        # wake-up detected
        diffs = np.diff(prs)
        max_jump = np.max(diffs)
        avg_jump = np.mean(diffs)
        if max_jump > avg_jump * 3:
            wake_type = 'explosive'
        else:
            wake_type = 'gradual'

    return 'good', f'Wake-up: {wake_type}'


# ============================================================
# D. Streamlit 前端
# ============================================================
def main():
    st.title("FeRAM Lab")
    st.markdown("---")

    with st.sidebar:
        st.header("Capacitor Area")
        L = st.number_input("length (µm)", value=50.0, step=10.0)
        W = st.number_input("width (µm)", value=50.0, step=10.0)
        area = L * W

        st.divider()
        st.subheader("Parameter Optimization")
        v_th = st.number_input("PUND threshold (V)", value=0.01, step=0.01, format="%.2f")

        st.caption("Advanced Correction")
        use_isolation = st.checkbox("Isolation", value=True)
        invert_pol = st.checkbox("I Invert", value=False)
        v_invert = st.checkbox("V Invert", value=False)
        use_comp = st.checkbox("Linear Comp", value=False)

        st.divider()
        plot_type = st.radio("Figure style", ["Line", "Scatter", "Both"])
        cmap_option = st.selectbox("Color type", ["viridis", "plasma", "rainbow", "coolwarm"])

        st.divider()
        debug_mode = st.checkbox("More Information", value=False)

        st.divider()
        api_key = st.text_input("Gemini API Key", type="password")

    uploaded_file = st.file_uploader(
        "Upload file (Support .zip, .7z, .xlsx, .csv)",
        type=["zip", "7z", "xlsx", "csv"])

    if not uploaded_file:
        st.info("Upload PUND raw data file\n\n**Supported：** .zip, .7z, .xlsx, .csv")
        return

    files_to_process = extract_archive(uploaded_file)
    if not files_to_process:
        st.error("未找到有效的資料檔案"); return

    pv_files = [(f, c) for f, c in files_to_process
                if not any(skip in f.lower() for skip in ['cycling', 'spgu', 'index'])]
    if not pv_files:
        st.warning("未找到 PV 資料檔"); return

    # ── 批次處理 ──
    results = []
    analyzers = {}
    debug_info = []
    bar = st.progress(0)
    status = st.empty()

    for i, (fname, fcontent) in enumerate(pv_files):
        status.text(f"分析: {fname}...")
        try:
            df = load_file(fcontent, fname)
            if df is not None and len(df) > 10:
                ana = FerroelectricAnalyzer(df, area, fname)
                res, err = ana.calculate(v_th=v_th, invert=invert_pol,
                                          linear_comp=use_comp, v_invert=v_invert)
                if res:
                    results.append(res)
                    analyzers[fname] = ana
                    debug_info.append({'file': fname, 'points': len(ana.V),
                        'dt(µs)': f"{np.median(np.diff(ana.T))*1e6:.2f}" if len(ana.T)>1 else "N/A",
                        'pulses': res.get('n_pulses', 0), 'status': '✓'})
                elif err:
                    debug_info.append({'file': fname, 'points': len(ana.V) if hasattr(ana,'V') else 0,
                        'dt(µs)': 'N/A', 'pulses': 0, 'status': f'✗ {err}'})
                    st.warning(f"⚠ {fname}: {err}")
        except Exception as e:
            debug_info.append({'file': fname, 'points': 0, 'dt(µs)': 'N/A',
                'pulses': 0, 'status': f'✗ {e}'})
        bar.progress((i + 1) / len(pv_files))

    status.text(f"Analysis completed ({len(results)}/{len(pv_files)})")
    bar.empty()

    if not results:
        st.error("所有檔案處理失敗"); return

    df_res = pd.DataFrame(results).sort_values("cycle_num").reset_index(drop=True)

    # ════════════════════════════════════════
    tab1, tab2, tab3 = st.tabs(["P-E Loops", "Endurance", "AI Analysis"])

    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("#### 選擇週期")
            labels = df_res['cycle_label'].tolist()
            sel = st.multiselect("顯示圖層", labels, default=labels)
        with col2:
            if sel:
                fig, ax = plt.subplots(figsize=(7, 5.5))
                cmap = plt.get_cmap(cmap_option)
                colors = cmap(np.linspace(0.1, 0.9, max(len(sel), 1)))
                for i, lbl in enumerate(sel):
                    row = df_res[df_res['cycle_label'] == lbl].iloc[0]
                    vd, pd_ = row['data_V'], row['data_P']
                    if "Line" in plot_type or "Both" in plot_type:
                        ax.plot(vd, pd_, label=lbl, lw=1.8, color=colors[i], alpha=0.85)
                    if "Scatter" in plot_type or "Both" in plot_type:
                        ax.scatter(vd, pd_, s=12, color=colors[i], alpha=0.5, edgecolors='none')
                ax.set_xlabel("Voltage (V)", fontsize=13)
                ax.set_ylabel("Polarization (µC/cm²)", fontsize=13)
                ax.axhline(0, c='gray', lw=0.5, ls='--')
                ax.axvline(0, c='gray', lw=0.5, ls='--')
                ax.set_title("P-E Hysteresis Loops", fontsize=14, fontweight='bold')
                ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
                fig.tight_layout(); st.pyplot(fig)
                img = io.BytesIO()
                fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
                st.download_button("Download chart", img, "pe_loops.png", "image/png")
                if st.checkbox("Show 2Pr / 2Vc table"):
                    st.dataframe(df_res[df_res['cycle_label'].isin(sel)][
                        ['cycle_label', '2Pr', '2Vc', 'filename']].round(4).reset_index(drop=True),
                        use_container_width=True)

    with tab2:
        if len(df_res) > 1:
            st.markdown("#### Endurance Trend")
            fig2, ax = plt.subplots(figsize=(10, 5))
            ax.semilogx(df_res['cycle_num'], df_res['2Pr'], 'o-', color='tab:blue', label='2Pr', lw=2, markersize=8)
            ax.set_xlabel("Cycles", fontsize=13); ax.set_ylabel("2Pr (µC/cm²)", color='tab:blue', fontsize=13)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax2 = ax.twinx()
            ax2.semilogx(df_res['cycle_num'], df_res['2Vc'], 's--', color='tab:red', label='2Vc', lw=1.5, markersize=6, alpha=0.7)
            ax2.set_ylabel("2Vc (V)", color='tab:red', fontsize=13)
            lines, l1 = ax.get_legend_handles_labels()
            lines2, l2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, l1 + l2, loc='best')
            fig2.tight_layout(); st.pyplot(fig2)
            csv = df_res[['cycle_label', 'cycle_num', '2Pr', '2Vc']].to_csv(index=False).encode()
            st.download_button("Download CSV", csv, "endurance.csv", "text/csv")

    with tab3:
        st.markdown("#### Gemini Assistant")
        if not api_key:
            st.info("請在側邊欄輸入 Gemini API Key。")
        else:
            if st.button("開始 AI 分析"):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    table = df_res[['cycle_label', '2Pr', '2Vc']].to_markdown(index=False)
                    prompt = f"分析 FeRAM 耐久度:\n{table}\n1.Wake-up 2.疲勞 3.機制 4.可靠度"
                    with st.spinner("分析中..."):
                        st.markdown(model.generate_content(prompt).text)
                except Exception as e:
                    st.error(str(e))

    # ════════════════════════════════════════
    # More Information
    # ════════════════════════════════════════
    if debug_mode:
        st.markdown("---")
        st.markdown("### More Information")

        dt1, dt2, dt3, dt4, dt5, dt6 = st.tabs([
            "Feature Table", "Trends", "Switching Transient",
            "NLS Fitting", "Non-switching Current", "VIT waveform"
        ])

        # Feature Table 
        with dt1:
            st.markdown("#### Physical Features per Cycle")
            feat_rows = []
            for r in sorted(results, key=lambda x: x['cycle_num']):
                a = r.get('advanced', {})
                feat_rows.append({
                    'Cycle': r['cycle_label'],
                    '2Pr (µC/cm²)': f"{a.get('2Pr', 0):.2f}",
                    '2Vc (V)': f"{a.get('2Vc', 0):.3f}",
                    'Pr+ (µC/cm²)': f"{a.get('Pr_pos', 0):.2f}",
                    'Pr- (µC/cm²)': f"{a.get('Pr_neg', 0):.2f}",
                    'Vc+ (V)': f"{a.get('Vc_pos', 0):.3f}",
                    'Vc- (V)': f"{a.get('Vc_neg', 0):.3f}",
                    'Imprint (V)': f"{a.get('imprint', 0):.3f}",
                    'P_sat (µC/cm²)': f"{a.get('P_sat', 0):.2f}",
                    'Squareness': f"{a.get('squareness', 0):.3f}",
                    'Pr_asym': f"{a.get('Pr_asym', 0):.3f}",
                    'I_sw_peak (µA)': f"{a.get('I_sw_peak', 0):.2f}",
                    't_peak (µs)': f"{a.get('t_peak', 0):.1f}",
                    'FWHM (µs)': f"{a.get('FWHM', 0):.1f}",
                    't_90 (µs)': f"{a.get('t_90', 0):.1f}",
                    'I_leak (µA)': f"{a.get('I_leak', 0):.2f}",
                    'I_tail (µA)': f"{a.get('I_tail', 0):.3f}",
                    'Loop Area (µJ/cm²)': f"{a.get('loop_area', 0):.1f}",
                    'NLS t₀ (µs)': f"{a.get('NLS_t0', 0):.2f}",
                    'NLS n': f"{a.get('NLS_n', 0):.2f}",
                    'NLS R²': f"{a.get('NLS_R2', 0):.3f}",
                })
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True)

            csv_feat = pd.DataFrame(feat_rows).to_csv(index=False).encode()
            st.download_button("Download Features CSV", csv_feat, "features.csv", "text/csv")

        # Trends 
        with dt2:
            if len(results) < 2:
                st.info("需要 ≥2 個 cycle")
            else:
                sorted_res = sorted(results, key=lambda x: x['cycle_num'])
                cycles = [r['cycle_num'] for r in sorted_res]
                advs = [r.get('advanced', {}) for r in sorted_res]

                plot_items = [
                    ('Imprint (V)', 'imprint', 'tab:purple'),
                    ('I_leak (µA)', 'I_leak', 'tab:red'),
                    ('Loop Area (µJ/cm²)', 'loop_area', 'tab:orange'),
                    ('Squareness', 'squareness', 'tab:green'),
                    ('I_sw_peak (µA)', 'I_sw_peak', 'tab:blue'),
                    ('FWHM (µs)', 'FWHM', 'tab:cyan'),
                ]

                fig,axes = plt.subplots(2, 3, figsize=(14, 8))
                axes = axes.flatten()
                for idx, (title, key, color) in enumerate(plot_items):
                    ax = axes[idx]
                    vals = [a.get(key, 0) for a in advs]
                    ax.semilogx(cycles, vals, 'o-', color=color, lw=2, markersize=7)
                    ax.set_xlabel('Cycles'); ax.set_ylabel(title)
                    ax.set_title(title, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                fig.suptitle('Physical Parameter Trends', fontsize=14, fontweight='bold')
                fig.tight_layout(); st.pyplot(fig)

                img = io.BytesIO()
                fig.savefig(img, format='png', dpi=200, bbox_inches='tight')
                st.download_button("Download Trends", img, "trends.png", "image/png")

        # Switching Transient
        with dt3:
            st.markdown("#### Switching Current I_ferro(t) per Cycle")
            sorted_res = sorted(results, key=lambda x: x['cycle_num'])
            cmap = plt.get_cmap(cmap_option)
            colors = cmap(np.linspace(0.1, 0.9, max(len(sorted_res), 1)))

            # 判斷 pair1 的有效極性（考慮 V Invert）
            if sorted_res:
                sample_V1 = sorted_res[0].get('V_1', np.array([0]))
                eff_pol_1 = 1 if np.mean(sample_V1) > 0 else -1
                if v_invert:
                    eff_pol_1 *= -1
                if eff_pol_1 > 0:
                    title1, title2 = 'P pulse (positive switching)', 'N pulse (negative switching)'
                    nsw_title1, nsw_title2 = 'U pulse (positive non-switching)', 'D pulse (negative non-switching)'
                else:
                    title1, title2 = 'N pulse (negative switching)', 'P pulse (positive switching)'
                    nsw_title1, nsw_title2 = 'D pulse (negative non-switching)', 'U pulse (positive non-switching)'
            else:
                title1, title2 = 'Half-loop 1', 'Half-loop 2'
                nsw_title1, nsw_title2 = 'Non-switching 1', 'Non-switching 2'

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for i, r in enumerate(sorted_res):
                dt_val = r.get('dt', 5e-7)
                I_f1 = r.get('I_ferro_1', np.array([]))
                I_f2 = r.get('I_ferro_2', np.array([]))
                if len(I_f1) > 0:
                    t1 = np.arange(len(I_f1)) * dt_val * 1e6
                    ax1.plot(t1, I_f1 * 1e6, color=colors[i], label=r['cycle_label'], lw=1.5, alpha=0.8)
                if len(I_f2) > 0:
                    t2 = np.arange(len(I_f2)) * dt_val * 1e6
                    ax2.plot(t2, I_f2 * 1e6, color=colors[i], label=r['cycle_label'], lw=1.5, alpha=0.8)

            ax1.set_xlabel('Time (µs)'); ax1.set_ylabel('I_ferro (µA)')
            ax1.set_title(title1, fontweight='bold')
            ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
            ax2.set_xlabel('Time (µs)'); ax2.set_ylabel('I_ferro (µA)')
            ax2.set_title(title2, fontweight='bold')
            ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
            fig.tight_layout(); st.pyplot(fig)

        # NLS Fitting
        with dt4:
            st.markdown("#### Nucleation-Limited Switching (KAI Model)")
            st.latex(r"Q(t) = 1 - \exp\left[-\left(\frac{t}{t_0}\right)^n\right]")

            sorted_res = sorted(results, key=lambda x: x['cycle_num'])

            fig, axes = plt.subplots(1, min(4, len(sorted_res)), figsize=(4 * min(4, len(sorted_res)), 4))
            if len(sorted_res) == 1: axes = [axes]
            show_idx = np.linspace(0, len(sorted_res) - 1, min(4, len(sorted_res))).astype(int)

            for plot_i, ri in enumerate(show_idx):
                r = sorted_res[ri]
                dt = r.get('dt', 5e-7)
                I_f1 = r.get('I_ferro_1', np.array([]))
                a = r.get('advanced', {})
                ax = axes[plot_i]

                if len(I_f1) > 10:
                    Q = np.cumsum(np.abs(I_f1) * dt)
                    if Q[-1] > 0:
                        Q_norm = Q / Q[-1]
                        t_arr = np.arange(len(Q_norm)) * dt * 1e6

                        ax.plot(t_arr, Q_norm, 'b-', lw=2, label='Data')

                        # Plot fit if available
                        t0 = a.get('NLS_t0', 0)
                        n = a.get('NLS_n', 0)
                        r2 = a.get('NLS_R2', 0)
                        if t0 > 0 and n > 0:
                            t_fit = np.arange(len(Q_norm)) * dt
                            Q_fit = 1 - np.exp(-(t_fit / (t0 * 1e-6)) ** n)
                            ax.plot(t_arr, Q_fit, 'r--', lw=1.5, label=f'KAI fit')
                            ax.set_title(f"{r['cycle_label']}\nt₀={t0:.1f}µs, n={n:.2f}, R²={r2:.3f}",
                                         fontsize=10)
                        else:
                            ax.set_title(f"{r['cycle_label']}\nFit failed", fontsize=10)

                ax.set_xlabel('Time (µs)'); ax.set_ylabel('Q/Q_total')
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.15)
            fig.suptitle('NLS Model Fitting', fontweight='bold')
            fig.tight_layout(); st.pyplot(fig)

            # NLS params trend
            if len(sorted_res) >= 3:
                st.markdown("#### NLS Parameters vs Cycle")
                cycles = [r['cycle_num'] for r in sorted_res]
                t0s = [r['advanced'].get('NLS_t0', 0) for r in sorted_res]
                ns = [r['advanced'].get('NLS_n', 0) for r in sorted_res]
                r2s = [r['advanced'].get('NLS_R2', 0) for r in sorted_res]

                fig2, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4))
                a1.semilogx(cycles, t0s, 'o-', color='tab:blue', lw=2)
                a1.set_ylabel('t₀ (µs)'); a1.set_title('Switching Time')
                a2.semilogx(cycles, ns, 's-', color='tab:red', lw=2)
                a2.set_ylabel('n'); a2.set_title('Avrami Index')
                a2.axhline(1, ls=':', color='gray', alpha=0.5)
                a2.axhline(2, ls=':', color='gray', alpha=0.5)
                a3.semilogx(cycles, r2s, 'D-', color='tab:green', lw=2)
                a3.set_ylabel('R²'); a3.set_title('Fit Quality')
                for a in [a1, a2, a3]:
                    a.set_xlabel('Cycles'); a.grid(True, alpha=0.3)
                fig2.tight_layout(); st.pyplot(fig2)

        # Non-switching Current
        with dt5:
            st.markdown("#### Non-switching Current — Pure Leakage")
            sorted_res = sorted(results, key=lambda x: x['cycle_num'])
            cmap = plt.get_cmap(cmap_option)
            colors = cmap(np.linspace(0.1, 0.9, max(len(sorted_res), 1)))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for i, r in enumerate(sorted_res):
                dt_val = r.get('dt', 5e-7)
                I_nsw1 = r.get('I_nsw1', np.array([]))
                I_nsw2 = r.get('I_nsw2', np.array([]))
                if len(I_nsw1) > 0:
                    t = np.arange(len(I_nsw1)) * dt_val * 1e6
                    ax1.plot(t, np.abs(I_nsw1) * 1e6, color=colors[i],
                             label=r['cycle_label'], lw=1.2, alpha=0.8)
                if len(I_nsw2) > 0:
                    t = np.arange(len(I_nsw2)) * dt_val * 1e6
                    ax2.plot(t, np.abs(I_nsw2) * 1e6, color=colors[i],
                             label=r['cycle_label'], lw=1.2, alpha=0.8)

            ax1.set_xlabel('Time (µs)'); ax1.set_ylabel('|I_nsw| (µA)')
            ax1.set_title(nsw_title1, fontweight='bold')
            ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
            ax2.set_xlabel('Time (µs)'); ax2.set_ylabel('|I_nsw| (µA)')
            ax2.set_title(nsw_title2, fontweight='bold')
            ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
            fig.tight_layout(); st.pyplot(fig)

        # VIT waveform
        with dt6:
            st.markdown("---")
            st.markdown("#### Data Point")
            st.dataframe(pd.DataFrame(debug_info), use_container_width=True)

            if analyzers:
                st.markdown("#### VIT Waveform")
                debug_file = st.selectbox("select file", list(analyzers.keys()))
                if debug_file and debug_file in analyzers:
                    ana = analyzers[debug_file]
                    fig_d, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                    t_us = ana.T * 1e6 if np.max(ana.T) < 1 else ana.T
                    t_label = "Time (µs)" if np.max(ana.T) < 1 else "Time"
                    ax_v.plot(t_us, ana.V, 'b-', lw=1.2)
                    ax_v.set_ylabel('Voltage (V)', color='blue', fontsize=12)
                    ax_v.axhline(v_th, color='red', ls=':', lw=0.8, alpha=0.5)
                    ax_v.axhline(-v_th, color='red', ls=':', lw=0.8, alpha=0.5)
                    ax_v.grid(True, alpha=0.3)

                    pulses_d = ana._detect_pulses(v_th)
                    p_pair_d, n_pair_d = ana._assign_pund(pulses_d)
                    for j, p in enumerate(pulses_d):
                        c = '#90EE9040' if p['pol'] > 0 else '#FFD70040'
                        ax_v.axvspan(t_us[p['s']], t_us[min(p['e'], len(t_us)-1)], alpha=0.3, color=c)

                    pund_labels = {}
                    if p_pair_d and n_pair_d:
                        eff = p_pair_d[0]['pol'] * (-1 if v_invert else 1)
                        if eff > 0:
                            pund_labels.update({id(p_pair_d[0]):'P', id(p_pair_d[1]):'U',
                                                id(n_pair_d[0]):'N', id(n_pair_d[1]):'D'})
                        else:
                            pund_labels.update({id(p_pair_d[0]):'N', id(p_pair_d[1]):'D',
                                                id(n_pair_d[0]):'P', id(n_pair_d[1]):'U'})

                    for j, p in enumerate(pulses_d):
                        mid_t = t_us[(p['s']+p['e'])//2]
                        lbl = pund_labels.get(id(p), f"P{j+1}")
                        y = ana.V[p['s']:p['e']+1].max()*0.8 if p['pol']>0 else ana.V[p['s']:p['e']+1].min()*0.8
                        ax_v.annotate(lbl, xy=(mid_t, y), fontsize=11, fontweight='bold',
                                      color='red', ha='center',
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

                    ax_i.plot(t_us, ana.I * 1e6, 'r-', lw=1)
                    ax_i.set_ylabel('Current (µA)', color='red', fontsize=12)
                    ax_i.set_xlabel(t_label, fontsize=12)
                    ax_i.grid(True, alpha=0.3)
                    fig_d.suptitle(debug_file, fontsize=12, fontweight='bold')
                    fig_d.tight_layout(); st.pyplot(fig_d)


if __name__ == "__main__":
    main()
