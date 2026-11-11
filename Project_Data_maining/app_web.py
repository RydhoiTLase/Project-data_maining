"""ClusterMhs v3 - Website Clustering Mahasiswa Tanjungpinang (K-Means).
Jalankan:  pip install streamlit pandas numpy altair   lalu   streamlit run app_web.py
Seluruh parameter model (centroid, scaler, titik data) sudah tertanam di file ini.
"""
import html
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ======================= PARAMETER MODEL (hasil Colab) =======================
M = {'akt': [[20.6, 8.7, 4.2, 3.6, 10.5], [18.9, 7.5, 3.4, 3.1, 11.0], [17.7, 8.2, 3.8, 3.2, 10.9]],
 'cap': 593250.0,
 'cent': [[1.671, 0.156], [-0.438, 0.845], [-0.192, -0.918]],
 'ipk': [3.55, 3.75, 3.24],
 'komp': [[0.039, 0.136, 0.048, 0.016, 0.336, 0.107, 0.175, 0.143],
          [0.028, 0.141, 0.049, 0.014, 0.474, 0.128, 0.108, 0.042],
          [0.012, 0.126, 0.066, 0.009, 0.422, 0.129, 0.173, 0.064]],
 'mean': [258614.24, 3.51],
 'n': [24, 64, 63],
 'names': {0: 'Pengeluaran Tinggi & Berprestasi', 1: 'Hemat & Berprestasi', 2: 'Hemat, Perlu Pendampingan'},
 'peng': [555000, 201000, 234000],
 'pts': [[295, 3.35, 2], [455, 3.61, 0], [359, 3.44, 0], [169, 2.89, 2], [333, 3.85, 1], [392, 3.32, 0],
         [202, 3.27, 2], [92, 3.2, 2], [190, 3.93, 1], [350, 3.32, 2], [65, 3.62, 1], [338, 3.16, 2],
         [436, 3.71, 0], [258, 3.7, 1], [330, 4.0, 1], [373, 3.58, 0], [128, 3.45, 2], [499, 4.0, 0],
         [86, 3.63, 1], [131, 3.77, 1], [124, 3.3, 2], [299, 3.3, 2], [315, 3.75, 1], [268, 3.41, 2],
         [127, 3.74, 1], [277, 3.36, 2], [74, 3.61, 1], [518, 3.34, 0], [94, 3.66, 1], [277, 3.41, 2],
         [130, 3.7, 1], [139, 3.57, 1], [478, 3.41, 0], [76, 3.72, 1], [245, 3.44, 2], [338, 3.29, 2],
         [204, 3.33, 2], [202, 3.7, 1], [276, 3.4, 2], [163, 3.9, 1], [237, 3.63, 1], [455, 3.59, 0],
         [374, 2.84, 2], [394, 3.5, 0], [146, 3.35, 2], [306, 3.79, 1], [289, 3.44, 2], [291, 3.39, 2],
         [138, 3.8, 1], [97, 3.33, 2], [219, 3.39, 2], [101, 3.38, 2], [294, 3.1, 2], [259, 2.96, 2],
         [214, 3.2, 2], [129, 2.92, 2], [440, 3.57, 0], [249, 3.23, 2], [228, 3.3, 2], [320, 3.47, 2],
         [151, 2.96, 2], [324, 3.42, 2], [388, 2.84, 2], [593, 3.0, 0], [378, 3.42, 0], [148, 3.7, 1],
         [270, 3.8, 1], [288, 3.39, 2], [95, 3.55, 1], [277, 2.98, 2], [277, 3.68, 1], [193, 3.65, 1],
         [159, 3.12, 2], [229, 3.64, 1], [342, 3.61, 0], [315, 3.25, 2], [422, 3.52, 0], [512, 3.53, 0],
         [118, 3.88, 1], [166, 3.9, 1], [185, 3.57, 1], [324, 3.48, 2], [230, 3.14, 2], [593, 3.8, 0],
         [135, 3.65, 1], [245, 3.41, 2], [215, 3.11, 2], [0, 4.0, 1], [335, 3.75, 1], [70, 3.8, 1],
         [70, 4.0, 1], [200, 3.35, 2], [157, 3.65, 1], [131, 4.0, 1], [340, 3.85, 1], [259, 3.93, 1],
         [290, 3.56, 1], [270, 3.0, 2], [210, 2.8, 2], [25, 3.97, 1], [593, 3.8, 0], [380, 3.75, 0],
         [241, 3.38, 2], [143, 3.88, 1], [15, 3.83, 1], [151, 3.4, 2], [170, 3.96, 1], [212, 3.13, 2],
         [338, 3.06, 2], [593, 3.8, 0], [331, 3.95, 1], [220, 3.4, 2], [121, 3.45, 2], [111, 3.6, 1],
         [139, 3.34, 2], [113, 3.06, 2], [593, 3.42, 0], [108, 3.25, 2], [91, 3.5, 1], [434, 2.94, 2],
         [133, 3.2, 2], [96, 3.69, 1], [117, 3.3, 2], [228, 3.42, 2], [270, 3.32, 2], [163, 3.52, 1],
         [143, 3.25, 2], [265, 3.8, 1], [235, 3.2, 2], [290, 3.8, 1], [295, 3.7, 1], [339, 3.8, 1],
         [298, 3.6, 1], [593, 3.54, 0], [593, 3.57, 0], [247, 3.63, 1], [334, 3.67, 1], [255, 3.75, 1],
         [300, 3.89, 1], [315, 3.28, 2], [294, 3.7, 1], [299, 3.5, 2], [303, 3.78, 1], [277, 3.67, 1],
         [287, 3.66, 1], [259, 3.83, 1], [280, 3.8, 1], [280, 3.98, 1], [210, 2.78, 2], [460, 3.42, 0],
         [280, 3.59, 1]],
 'std': [130621.16, 0.29]}
NAMA = M["names"]
K = len(NAMA)
COL = ["#5B8DEF", "#F2994A", "#9B7BEA", "#4CC2A8", "#E86F8A"][:K]
KAT = ["Bensin", "Transportasi", "Pulsa/Langganan", "Akademik", "Makan", "Minum/Ngopi", "Jajan", "Lainnya"]
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
AKT = ["Kuliah", "Belajar mandiri", "Organisasi", "Bekerja", "Kegiatan lain"]
SARAN_NAMA = {
    "Hemat & Berprestasi": "Pola keuangan dan akademik sudah baik. Pertahankan, dan Anda bisa menjadi mentor sebaya.",
    "Hemat, Perlu Pendampingan": "Pengeluaran moderat namun IPK di bawah rata-rata. Tambah jam belajar dan cari pendampingan akademik.",
    "Pengeluaran Tinggi & Berprestasi": "Prestasi baik, tetapi pengeluaran tinggi. Susun anggaran mingguan dan kurangi jajan/kategori lainnya.",
    "Pengeluaran Tinggi, Prestasi Menurun": "Pengeluaran tinggi dengan IPK di bawah rata-rata. Prioritaskan manajemen keuangan dan waktu belajar.",
}


class _Saran(dict):
    def get(self, c, default=""):
        n = NAMA.get(c, "")
        return next((v for k, v in SARAN_NAMA.items() if n.startswith(k)), default)


SARAN = _Saran()
DF = pd.DataFrame(M["pts"], columns=["pengeluaran", "ipk", "cluster"])
DF["nama"] = DF["cluster"].map(NAMA)
CENT = pd.DataFrame({
    "pengeluaran": [c[0] * M["std"][0] / 1000 + M["mean"][0] / 1000 for c in M["cent"]],
    "ipk": [c[1] * M["std"][1] + M["mean"][1] for c in M["cent"]],
    "nama": [NAMA[i] for i in range(K)]})


def klasifikasi(total, ipk):
    """Nearest-centroid pada skala z-score (identik dengan K-Means.predict)."""
    z = np.array([(min(total, M["cap"]) - M["mean"][0]) / M["std"][0],
                  (ipk - M["mean"][1]) / M["std"][1]])
    d = np.linalg.norm(np.array(M["cent"]) - z, axis=1)
    w = 1 / (d + 1e-6)
    return int(d.argmin()), w / w.sum()


def persentil(v, kolom):
    return int(round((DF[kolom] <= v).mean() * 100))


def rp(n):
    return "Rp" + f"{int(round(n)):,}".replace(",", ".")


METRIK = {"Model Utama (Rp × IPK)": dict(wcss=120.95, sil=0.3984, db=0.8672, ch=110.77),
          "Pembanding (7 fitur)": dict(wcss=808.01, sil=0.1368, db=2.1367, ch=22.8)}
JADWAL = [("Business Understanding & penyusunan proposal", "Rydhoi Trimaniel Lase", "2026-09-08", "2026-09-19"),
          ("Data Understanding", "Riki Andika Saputra", "2026-09-15", "2026-09-18"),
          ("EDA: data deskriptif & eksplorasi", "Riki Andika Saputra", "2026-09-22", "2026-09-25"),
          ("EDA: visualisasi data", "Riki Andika Saputra", "2026-09-29", "2026-10-02"),
          ("Data Selection, Transformation & Feature Engineering", "Ardiansyah Rizki Khairi Ali", "2026-10-13", "2026-10-16"),
          ("Pemodelan K-Means", "Aldi Saputra", "2026-10-27", "2026-10-30"),
          ("Evaluasi & pengukuran model", "Ikhbal Maulana", "2026-11-10", "2026-11-13"),
          ("Penyusunan laporan & video presentasi final", "Rydhoi Trimaniel Lase", "2026-11-17", "2026-11-27")]

# ================================ TAMPILAN ===================================
st.set_page_config(page_title="ClusterMhs", page_icon="📊", layout="wide")
st.markdown("""<style>
:root{--ac:#5B8DEF;--ac2:#E4ECFD;--tx:#1E2A44;--mut:#6B7A99;--bd:#E3EAF7}
.stApp{background:#EEF3FC;color:var(--tx)}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.6rem;max-width:1250px}
section[data-testid="stSidebar"]{background:#CFE0FB;border-radius:0 24px 24px 0}
section[data-testid="stSidebar"] *{color:#1E2A44!important}
div[role="radiogroup"] label{background:transparent;border-radius:14px;padding:10px 14px;margin-bottom:4px;width:100%}
div[role="radiogroup"] label:has(input:checked){background:#fff;box-shadow:0 6px 24px rgba(91,141,239,.15)}
div[role="radiogroup"] label>div:first-child{display:none}
h1,h2,h3,p,label,span{color:var(--tx)}
.card{background:#fff;border-radius:20px;padding:18px 20px;box-shadow:0 6px 24px rgba(91,141,239,.12);border:1px solid var(--bd);margin-bottom:14px}
.card h4{margin:0 0 8px;font-size:14px;color:var(--tx)}
.stat .ic{width:38px;height:38px;border-radius:12px;background:var(--ac2);display:grid;place-items:center;font-size:18px}
.stat b{display:block;font-size:26px;margin-top:8px;color:var(--tx)}.stat span{color:var(--mut);font-size:12px}
.pill{display:inline-block;padding:3px 12px;border-radius:99px;font-size:12px;font-weight:600;color:#fff!important}
.bar{height:8px;border-radius:9px;background:var(--ac2);overflow:hidden;margin:3px 0 9px}.bar i{display:block;height:100%}
.row{display:flex;justify-content:space-between;font-size:13px;color:var(--tx)}
.stack{display:flex;height:10px;border-radius:9px;overflow:hidden;margin:6px 0}
.mut{color:var(--mut)!important;font-size:13px}
.stButton>button,.stFormSubmitButton>button,.stDownloadButton>button{background:var(--ac);color:#fff;border:0;border-radius:12px;padding:.5rem 1.4rem;font-weight:600}
.stNumberInput input{background:#F5F8FE!important;color:var(--tx)!important;border-radius:12px}
</style>""", unsafe_allow_html=True)


st.markdown('<style>\n.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;\nbackground:url("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20width%3D%27900%27%20height%3D%27700%27%20viewBox%3D%270%200%20900%20700%27%20fill%3D%27none%27%20stroke%3D%27%235B8DEF%27%20stroke-opacity%3D%27.28%27%20stroke-width%3D%271.5%27%3E%3Cpath%20d%3D%27M520%2080L640%20150L770%2090L850%20210L700%20270L640%20150M700%20270L560%20340L430%20260L520%2080M560%20340L640%20470L780%20420L850%20210M430%20260L330%20380L460%20480L560%20340M640%20470L600%20600L720%20640L780%20420M330%20380L200%20340%27/%3E%3Cg%20fill%3D%27%235B8DEF%27%20fill-opacity%3D%27.35%27%20stroke%3D%27none%27%3E%3Ccircle%20cx%3D%27520%27%20cy%3D%2780%27%20r%3D%278%27/%3E%3Ccircle%20cx%3D%27640%27%20cy%3D%27150%27%20r%3D%2711%27/%3E%3Ccircle%20cx%3D%27770%27%20cy%3D%2790%27%20r%3D%277%27/%3E%3Ccircle%20cx%3D%27850%27%20cy%3D%27210%27%20r%3D%279%27/%3E%3Ccircle%20cx%3D%27700%27%20cy%3D%27270%27%20r%3D%2712%27/%3E%3Ccircle%20cx%3D%27560%27%20cy%3D%27340%27%20r%3D%279%27/%3E%3Ccircle%20cx%3D%27430%27%20cy%3D%27260%27%20r%3D%277%27/%3E%3Ccircle%20cx%3D%27330%27%20cy%3D%27380%27%20r%3D%278%27/%3E%3Ccircle%20cx%3D%27460%27%20cy%3D%27480%27%20r%3D%277%27/%3E%3Ccircle%20cx%3D%27640%27%20cy%3D%27470%27%20r%3D%2710%27/%3E%3Ccircle%20cx%3D%27780%27%20cy%3D%27420%27%20r%3D%278%27/%3E%3Ccircle%20cx%3D%27600%27%20cy%3D%27600%27%20r%3D%277%27/%3E%3Ccircle%20cx%3D%27720%27%20cy%3D%27640%27%20r%3D%279%27/%3E%3Ccircle%20cx%3D%27200%27%20cy%3D%27340%27%20r%3D%276%27/%3E%3C/g%3E%3C/svg%3E") no-repeat right -80px top 10px/780px,url("data:image/svg+xml,%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20width%3D%27800%27%20height%3D%27600%27%20viewBox%3D%270%200%20800%20600%27%20fill%3D%27none%27%20stroke%3D%27%239B7BEA%27%20stroke-opacity%3D%27.30%27%20stroke-width%3D%271.6%27%3E%3Cpath%20d%3D%27M0%20420H140V360H280V300H400M0%20500H200V450H340V390H470M0%20340H80V270H210M120%20600V540H260V500H380M300%20600V560H520V470%27/%3E%3Crect%20x%3D%27400%27%20y%3D%27270%27%20width%3D%2760%27%20height%3D%2760%27%20rx%3D%278%27/%3E%3Crect%20x%3D%27470%27%20y%3D%27360%27%20width%3D%2746%27%20height%3D%2746%27%20rx%3D%277%27/%3E%3Crect%20x%3D%27210%27%20y%3D%27240%27%20width%3D%2740%27%20height%3D%2740%27%20rx%3D%276%27/%3E%3Cg%20fill%3D%27%239B7BEA%27%20fill-opacity%3D%27.35%27%20stroke%3D%27none%27%3E%3Ccircle%20cx%3D%27140%27%20cy%3D%27360%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27280%27%20cy%3D%27300%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27340%27%20cy%3D%27390%27%20r%3D%276%27/%3E%3Ccircle%20cx%3D%27520%27%20cy%3D%27470%27%20r%3D%277%27/%3E%3Ccircle%20cx%3D%2780%27%20cy%3D%27270%27%20r%3D%276%27/%3E%3C/g%3E%3C/svg%3E") no-repeat left -60px bottom -30px/640px}\n.block-container{position:relative;z-index:1}\n.blob{position:fixed;border-radius:50%;filter:blur(70px);opacity:.38;z-index:0;pointer-events:none;animation:fl 16s ease-in-out infinite alternate}\n.b1{width:340px;height:340px;background:#9DBBFA;top:-80px;left:30%}.b2{width:300px;height:300px;background:#C9B8F7;bottom:-60px;right:8%;animation-delay:-6s}\n@keyframes fl{to{transform:translate(40px,30px) scale(1.15)}}\n@keyframes up{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}\n@keyframes pulse{50%{opacity:.35;transform:scale(1.4)}}\n.card{animation:up .55s ease both;transition:transform .25s,box-shadow .25s}\n.card:hover{transform:translateY(-3px);box-shadow:0 12px 32px rgba(91,141,239,.22)}\n.hero{background:linear-gradient(120deg,#5B8DEF,#86A9F6 55%,#B6A3F4);border-radius:28px;padding:30px 34px;display:flex;justify-content:space-between;align-items:center;gap:20px;overflow:hidden;box-shadow:0 14px 40px rgba(91,141,239,.35);margin-bottom:16px;animation:up .6s ease both}\n.hero *{color:#fff!important}.hero h1{font-size:28px;line-height:1.25;margin:10px 0 8px;max-width:560px}.hero p{margin:0;opacity:.92;max-width:520px}\n.tag{background:rgba(255,255,255,.22);border-radius:99px;padding:4px 14px;font-size:12px;font-weight:600}\n.hero circle.n{animation:pulse 2.4s ease-in-out infinite;transform-box:fill-box;transform-origin:center}\n.ins{display:flex;gap:12px;align-items:center}.ins .ic{flex:none}.ins b{font-size:15px}\n.step{display:flex;gap:14px;align-items:flex-start}.step .no{flex:none;width:34px;height:34px;border-radius:50%;background:var(--ac);color:#fff;display:grid;place-items:center;font-weight:700}\n.chip{display:inline-block;padding:2px 10px;border-radius:8px;font-size:12px;font-weight:600;background:var(--ac2);color:var(--tx)}\nbutton[kind="secondary"]{border-radius:12px!important;border:1px solid #D5E1F8!important;background:#fff!important;color:#1E2A44!important}\nbutton[kind="secondary"]:hover{border-color:#5B8DEF!important;color:#5B8DEF!important}\nbutton[data-baseweb="tab"]{font-weight:600}\n@media(max-width:800px){.hero svg{display:none}.hero h1{font-size:22px}}\n</style><div class="blob b1"></div><div class="blob b2"></div>', unsafe_allow_html=True)


st.markdown('<style>.rb{position:fixed;z-index:0;pointer-events:none}.rb1{right:18px;bottom:10px;animation:bob 3.2s ease-in-out infinite}.rb2{left:0;bottom:6px;transform:scale(.6);transform-origin:left bottom;animation:walk 46s linear infinite}.rb2 .wob{animation:bob .7s ease-in-out infinite}.rb3{left:0;top:84px;animation:fly 52s linear infinite}.rb3 .wob{animation:bob 2.4s ease-in-out infinite}@keyframes bob{50%{transform:translateY(-9px)}}@keyframes walk{from{translate:-160px 0}to{translate:100vw 0}}@keyframes fly{from{translate:100vw 0}to{translate:-160px 0}}@keyframes blink{0%,92%,100%{transform:scaleY(1)}96%{transform:scaleY(.08)}}@keyframes ant{50%{opacity:.25}}@keyframes wave{0%,100%{transform:rotate(0)}50%{transform:rotate(-38deg)}}@keyframes prop{50%{transform:scaleX(.45)}}.rb .eyes{transform-box:fill-box;transform-origin:center;animation:blink 4.5s infinite}.rb .ant{animation:ant 1.2s infinite}.rb .arm{transform-box:fill-box;transform-origin:50% 6%;animation:wave 1.6s ease-in-out infinite}.rb .prop{transform-box:fill-box;transform-origin:center;animation:prop .25s linear infinite}.bub{position:absolute;right:112px;top:-4px;background:#fff;border:2px solid #5B8DEF;border-radius:14px 14px 2px 14px;padding:5px 10px;font:600 12px sans-serif;color:#1E2A44;white-space:nowrap;animation:bubble 9s ease-in-out infinite}@keyframes bubble{0%,8%{opacity:0;transform:scale(.7)}14%,42%{opacity:1;transform:none}50%,100%{opacity:0}}@media(max-width:800px){.rb1,.rb3{display:none}}@media(prefers-reduced-motion:reduce){.rb,.rb *{animation:none!important}}</style><div class="rb rb1"><div class="bub">Halo! Aku Klaster-Bot 🤖</div><svg viewBox="0 0 120 160" width="130" height="173" fill="none"><line x1="60" y1="8" x2="60" y2="24" stroke="#5B8DEF" stroke-width="3"/><circle class="ant" cx="60" cy="8" r="6" fill="#F2994A"/><rect x="28" y="24" width="64" height="48" rx="16" fill="#fff" stroke="#5B8DEF" stroke-width="3"/><g class="eyes"><circle cx="46" cy="46" r="7" fill="#1E2A44"/><circle cx="74" cy="46" r="7" fill="#1E2A44"/><circle cx="48" cy="44" r="2" fill="#fff"/><circle cx="76" cy="44" r="2" fill="#fff"/></g><path d="M50 60Q60 67 70 60" stroke="#5B8DEF" stroke-width="3" stroke-linecap="round"/><rect x="38" y="78" width="44" height="46" rx="12" fill="#CFE0FB" stroke="#5B8DEF" stroke-width="3"/><rect x="46" y="88" width="28" height="16" rx="4" fill="#fff"/><rect x="50" y="97" width="4" height="4" fill="#5B8DEF"/><rect x="58" y="93" width="4" height="8" fill="#9B7BEA"/><rect x="66" y="90" width="4" height="11" fill="#5B8DEF"/><circle cx="52" cy="113" r="3" fill="#F2994A"/><circle cx="68" cy="113" r="3" fill="#4CC2A8"/><rect x="14" y="82" width="20" height="9" rx="4.5" fill="#9DBBFA" stroke="#5B8DEF" stroke-width="2"/><g class="arm"><rect x="82" y="64" width="9" height="28" rx="4.5" fill="#9DBBFA" stroke="#5B8DEF" stroke-width="2"/></g><rect x="44" y="126" width="12" height="24" rx="5" fill="#9DBBFA" stroke="#5B8DEF" stroke-width="2"/><rect x="64" y="126" width="12" height="24" rx="5" fill="#9DBBFA" stroke="#5B8DEF" stroke-width="2"/></svg></div><div class="rb rb2"><div class="wob"><svg viewBox="0 0 120 160" width="130" height="173" fill="none"><line x1="60" y1="8" x2="60" y2="24" stroke="#5B8DEF" stroke-width="3"/><circle class="ant" cx="60" cy="8" r="6" fill="#F2994A"/><rect x="28" y="24" width="64" height="48" rx="16" fill="#fff" stroke="#5B8DEF" stroke-width="3"/><g class="eyes"><circle cx="46" cy="46" r="7" fill="#1E2A44"/><circle cx="74" cy="46" r="7" fill="#1E2A44"/><circle cx="48" cy="44" r="2" fill="#fff"/><circle cx="76" cy="44" r="2" fill="#fff"/></g><path d="M50 60Q60 67 70 60" stroke="#5B8DEF" stroke-width="3" stroke-linecap="round"/><rect x="38" y="78" width="44" height="46" rx="12" fill="#CFE0FB" stroke="#5B8DEF" stroke-width="3"/><rect x="46" y="88" width="28" height="16" rx="4" fill="#fff"/><rect x="50" y="97" width="4" height="4" fill="#5B8DEF"/><rect x="58" y="93" width="4" height="8" fill="#9B7BEA"/><rect x="66" y="90" width="4" height="11" fill="#5B8DEF"/><circle cx="52" cy="113" r="3" fill="#F2994A"/><circle cx="68" cy="113" r="3" fill="#4CC2A8"/><rect x="14" y="82" width="20" height="9" rx="4.5" fill="#9DBBFA" stroke="#5B8DEF" stroke-width="2"/><g class="arm"><rect x="82" y="64" width="9" height="28" rx="4.5" fill="#9DBBFA" stroke="#5B8DEF" stroke-width="2"/></g><rect x="44" y="126" width="12" height="24" rx="5" fill="#9DBBFA" stroke="#5B8DEF" stroke-width="2"/><rect x="64" y="126" width="12" height="24" rx="5" fill="#9DBBFA" stroke="#5B8DEF" stroke-width="2"/></svg></div></div><div class="rb rb3"><div class="wob"><svg viewBox="0 0 90 70" width="86" height="67" fill="none"><ellipse class="prop" cx="45" cy="8" rx="30" ry="4" fill="#9B7BEA" fill-opacity=".6"/><line x1="45" y1="8" x2="45" y2="20" stroke="#9B7BEA" stroke-width="3"/><rect x="18" y="20" width="54" height="34" rx="14" fill="#fff" stroke="#9B7BEA" stroke-width="3"/><g class="eyes"><circle cx="35" cy="37" r="5.5" fill="#1E2A44"/><circle cx="55" cy="37" r="5.5" fill="#1E2A44"/></g><path d="M38 46Q45 51 52 46" stroke="#9B7BEA" stroke-width="2.5" stroke-linecap="round"/><path d="M30 54L26 64M60 54L64 64" stroke="#9B7BEA" stroke-width="3" stroke-linecap="round"/></svg></div></div>', unsafe_allow_html=True)


# ======================= LAPISAN VISUAL TAMBAHAN: MODERN TECH =======================
st.markdown("""<style>
:root{
  --tech-ink:#101b2d;
  --tech-navy:#14243a;
  --tech-steel:#263d59;
  --tech-blue:#2878d0;
  --tech-cyan:#20b9cf;
  --tech-line:#d4deea;
}
.stApp{
  background:linear-gradient(135deg,#edf2f7 0%,#e7edf5 52%,#f4f7fb 100%);
}
.block-container{max-width:1320px;padding-top:1.15rem;padding-bottom:2.8rem}
/* Hero dibuat lebih tegas dengan warna navy dan cyan. */
.hero{
  background:linear-gradient(118deg,#101b2d 0%,#183858 58%,#087c91 100%);
  border:1px solid rgba(255,255,255,.18);
  box-shadow:0 16px 42px rgba(16,35,58,.24);
  position:relative;
}
.hero::after{
  content:"";position:absolute;width:230px;height:230px;right:-72px;top:-112px;
  border:1px solid rgba(255,255,255,.23);border-radius:36px;transform:rotate(34deg);
  box-shadow:0 0 0 18px rgba(255,255,255,.035),0 0 0 38px rgba(255,255,255,.025);
  pointer-events:none;
}
.hero::before{
  content:"CLUSTER ANALYTICS / DATA LAB";position:absolute;right:28px;bottom:17px;
  color:rgba(255,255,255,.58);font:700 9px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;
  letter-spacing:2px;pointer-events:none;
}
/* Kartu tetap terang agar tabel, teks, dan visualisasi mudah dibaca. */
.card{
  border:1px solid var(--tech-line);border-top:2px solid #b9d4ef;
  border-radius:16px;box-shadow:0 7px 22px rgba(20,39,64,.075);
  position:relative;overflow:hidden;
}
.card::after{
  content:"";position:absolute;top:0;right:0;width:72px;height:72px;
  background:radial-gradient(circle at top right,rgba(32,185,207,.10),transparent 72%);
  pointer-events:none;
}
.card:hover{box-shadow:0 13px 30px rgba(20,39,64,.13);border-color:#b9cde2}
/* Sidebar bernuansa navy, tetapi menu tetap kontras dan terbaca. */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#101b2d 0%,#172d47 58%,#1b3852 100%);
  border-right:1px solid rgba(255,255,255,.08);
  box-shadow:8px 0 28px rgba(13,27,45,.10);
}
section[data-testid="stSidebar"] *{color:#e8f1fb!important}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#aebfd2!important}
section[data-testid="stSidebar"] div[role="radiogroup"] label{
  border:1px solid transparent;transition:all .2s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  background:rgba(255,255,255,.075);border-color:rgba(144,190,230,.2);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
  background:linear-gradient(100deg,rgba(40,120,208,.32),rgba(32,185,207,.13));
  border:1px solid rgba(111,190,229,.35);
  box-shadow:inset 3px 0 0 #35c3d5,0 5px 18px rgba(0,0,0,.12);
}
.sidebar-status{
  display:flex;align-items:center;justify-content:space-between;gap:8px;
  padding:9px 11px;margin:8px 0 14px;border-radius:11px;
  background:rgba(255,255,255,.075);border:1px solid rgba(177,209,239,.17);
  color:#c9d8e8;font:700 10px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;
  letter-spacing:1px;
}
.sidebar-status .online{color:#73e0c2!important}
.sidebar-status .dot{display:inline-block;width:7px;height:7px;margin-right:7px;
  border-radius:50%;background:#43d6ad;box-shadow:0 0 10px rgba(67,214,173,.8)}
/* Panel data, metrik, input, tombol, dan tab. */
[data-testid="stMetric"]{
  background:linear-gradient(145deg,#fff,#f7faff);padding:12px 14px;
  border:1px solid #dce5ef;border-radius:14px;
}
[data-testid="stMetricLabel"]{color:#64758b!important}
[data-testid="stMetricValue"]{color:#14243a!important;font-weight:750}
.stButton>button,.stFormSubmitButton>button,.stDownloadButton>button{
  background:linear-gradient(110deg,#1d5fa8,#287fc4 68%,#159db0);
  border:1px solid rgba(20,75,125,.15);box-shadow:0 5px 13px rgba(30,100,165,.16);
  transition:transform .18s ease,box-shadow .18s ease,filter .18s ease;
}
.stButton>button:hover,.stFormSubmitButton>button:hover,.stDownloadButton>button:hover{
  filter:brightness(1.06);transform:translateY(-1px);
  box-shadow:0 8px 17px rgba(30,100,165,.23);
}
.stNumberInput input,.stTextInput input,.stTextArea textarea{
  border:1px solid #d5e0ec!important;border-radius:10px!important;
}
[data-baseweb="select"]>div{border-radius:10px!important;border-color:#d5e0ec!important}
button[data-baseweb="tab"]{color:#52657b!important}
button[data-baseweb="tab"][aria-selected="true"]{
  color:#155b91!important;border-bottom-color:#20a9c0!important;
}
[data-testid="stDataFrame"]{border:1px solid #dce5ef;border-radius:12px;overflow:hidden}
.tech-ribbon{
  display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;
  margin:0 0 15px;padding:9px 14px;border:1px solid #d4dfeb;border-radius:12px;
  background:linear-gradient(100deg,#fff 0%,#f6f9fd 70%,#eef8fa 100%);
  box-shadow:0 4px 14px rgba(20,39,64,.045);
  color:#52657b;font:700 10px/1.3 ui-monospace,SFMono-Regular,Consolas,monospace;
  letter-spacing:1.05px;
}
.tech-ribbon .brand{color:#183858;font-weight:800}
.tech-ribbon .state{color:#178d79}
.tech-ribbon .state::before{content:"";display:inline-block;width:7px;height:7px;
  margin:0 7px 0 0;border-radius:50%;background:#24b995;box-shadow:0 0 8px rgba(36,185,149,.55)}
.tech-ribbon .meta{color:#708198;font-weight:600;letter-spacing:.7px}
/* Kurangi gerakan otomatis bagi pengguna yang memilih reduced motion. */
@media(prefers-reduced-motion:reduce){
  .card,.hero,.stButton>button,.stFormSubmitButton>button,.stDownloadButton>button{animation:none!important;transition:none!important}
}
@media(max-width:800px){
  .block-container{padding-top:.8rem}
  .tech-ribbon{font-size:9px}
  .hero::before{display:none}
}
</style>""", unsafe_allow_html=True)


def card(inner, extra=""):
    st.markdown(f'<div class="card" {extra}>{inner}</div>', unsafe_allow_html=True)


def pill(c):
    return f'<span class="pill" style="background:{COL[c]}">Cluster {c}</span>'


def judul(t, s):
    st.markdown(f"<h1 style='margin:0;font-size:26px'>{t}</h1><p class='mut' style='margin:2px 0 16px'>{s}</p>",
                unsafe_allow_html=True)


def peta(df, user=None, tinggi=330, sim=None):
    skala = alt.Scale(domain=[NAMA[i] for i in range(K)], range=COL)
    pts = alt.Chart(df).mark_circle(size=70, opacity=.78).encode(
        x=alt.X("pengeluaran:Q", title="Pengeluaran per minggu (ribu Rp)"),
        y=alt.Y("ipk:Q", title="IPK", scale=alt.Scale(zero=False)),
        color=alt.Color("nama:N", scale=skala, legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["nama", "pengeluaran", "ipk"])
    ctr = alt.Chart(CENT).mark_point(shape="diamond", size=260, filled=True, color="#F5C518",
                                     stroke="#333", strokeWidth=1).encode(x="pengeluaran:Q", y="ipk:Q",
                                                                          tooltip=["nama"])
    layers = [pts, ctr]
    if user:
        u = pd.DataFrame({"pengeluaran": [min(user[0], M["cap"]) / 1000], "ipk": [user[1]], "n": ["Anda"]})
        layers.append(alt.Chart(u).mark_point(shape="triangle-up", size=330, filled=True, color="#E5484D",
                                              stroke="#fff", strokeWidth=2).encode(x="pengeluaran:Q", y="ipk:Q",
                                                                                   tooltip=["n"]))
    if sim:
        (a, b) = sim
        L = pd.DataFrame({"x": [min(a[0], M["cap"]) / 1000, min(b[0], M["cap"]) / 1000], "y": [a[1], b[1]]})
        layers.append(alt.Chart(L).mark_line(color="#E5484D", strokeDash=[6, 4], strokeWidth=2).encode(x="x:Q", y="y:Q"))
        layers.append(alt.Chart(L.iloc[[1]]).mark_point(shape="cross", size=380, color="#16A34A", strokeWidth=4)
                      .encode(x="x:Q", y="y:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=tinggi).configure_view(strokeWidth=0)
                    .configure(background="transparent"), width="stretch")


# ================================= FUNGSI BANTU ==============================
PRESET = {"Hemat": (3.8, [15000, 20000, 15000, 20000, 15000, 25000, 20000], [18, 10, 4, 0, 10]),
          "Sedang": (3.3, [35000, 40000, 35000, 30000, 45000, 30000, 35000], [18, 6, 3, 2, 10]),
          "Boros": (3.6, [80000, 90000, 70000, 85000, 100000, 120000, 110000], [20, 8, 4, 3, 10])}


def isi(n):
    i, d, j = PRESET[n]
    st.session_state["ipk"] = i
    for k in range(7):
        st.session_state[f"d{k}"] = d[k]
    for k in range(5):
        st.session_state[f"a{k}"] = float(j[k])
    st.toast(f"Contoh “{n}” dimuat", icon="✨")


def keyakinan(share):
    a, b = np.sort(share)[::-1][:2]
    return ("Tinggi", "#16A34A") if a - b > .4 else ("Sedang", "#F2994A") if a - b > .2 else ("Dekat batas cluster", "#E5484D")


def butuh(target, tot, ipk):
    o_ipk = next((round(float(x), 2) for x in np.arange(ipk, 4.001, .01) if klasifikasi(tot, x)[0] == target), None)
    o_p = next((int(t) for t in np.arange(tot, -1, -5000) if klasifikasi(t, ipk)[0] == target), None)
    return o_ipk, o_p


HERO_SVG = """<svg width="300" height="170" viewBox="0 0 300 170" fill="none"><rect x="20" y="14" width="210" height="136" rx="16" fill="#fff" fill-opacity=".2" stroke="#fff" stroke-opacity=".7"/>
<rect x="38" y="104" width="22" height="30" rx="5" fill="#fff" fill-opacity=".85"/><rect x="70" y="80" width="22" height="54" rx="5" fill="#fff" fill-opacity=".85"/><rect x="102" y="92" width="22" height="42" rx="5" fill="#fff" fill-opacity=".85"/><rect x="134" y="58" width="22" height="76" rx="5" fill="#fff" fill-opacity=".85"/>
<path d="M38 70L84 48L122 62L170 30L214 40" stroke="#fff" stroke-width="3" stroke-linecap="round"/><circle class="n" cx="84" cy="48" r="5" fill="#fff"/><circle class="n" cx="170" cy="30" r="5" fill="#fff"/>
<circle cx="262" cy="54" r="22" fill="#fff" fill-opacity=".25" stroke="#fff"/><circle class="n" cx="262" cy="54" r="6" fill="#fff"/><circle cx="270" cy="124" r="16" fill="#fff" fill-opacity=".25" stroke="#fff"/><circle class="n" cx="270" cy="124" r="5" fill="#fff"/>
<path d="M230 70L252 62M230 110L256 120" stroke="#fff" stroke-width="2" stroke-dasharray="4 4"/></svg>"""

# ================================= NAVIGASI ==================================
MENU = {"🏠  Dashboard": "dash", "✨  Cek Cluster Saya": "cek", "🎯  Simulasi What-If": "sim",
        "🧩  Profil & Bandingkan": "profil", "🔎  Eksplorasi Data": "eks", "📘  Tentang Metode": "info"}
st.sidebar.markdown("### ◎ ClusterMhs")
st.sidebar.caption("Clustering Mahasiswa Tanjungpinang")
st.sidebar.markdown(
    '<div class="sidebar-status"><span><i class="dot"></i>MODEL ENGINE</span><span class="online">READY / K-MEANS</span></div>',
    unsafe_allow_html=True,
)
hal = MENU[st.sidebar.radio("Menu", list(MENU), label_visibility="collapsed")]
with st.sidebar.expander("💡 Panduan singkat"):
    st.markdown("1. **Cek Cluster Saya**: isi IPK & pengeluaran.\n2. **Simulasi**: coba ubah kebiasaan.\n"
                "3. **Profil**: pahami tiap kelompok.\n\n*Data bersifat anonim dan tidak dikirim ke server.*")
st.sidebar.caption("K-Means · Pengeluaran × IPK\n\nUniversitas Maritim Raja Ali Haji")
st.session_state.setdefault("hist", [])
st.session_state.setdefault("res", None)
if "ipk" not in st.session_state:
    isi("Sedang")

st.markdown(
    f"<div class='tech-ribbon'><span class='brand'>CLUSTERMHS <span style='color:#20a9c0'>/</span> ANALYTICS CONSOLE</span>"
    f"<span class='meta'>K-MEANS · {K} CLUSTERS · {len(DF)} RESPONDEN</span>"
    f"<span class='state'>DATASET LOADED</span></div>",
    unsafe_allow_html=True,
)

# ================================ DASHBOARD ==================================
if hal == "dash":
    st.markdown(f"""<div class="hero"><div><span class="tag">⚡ K-Means · Data Mining</span>
<h1>Kenali pola keuangan & prestasi akademik mahasiswa Tanjungpinang</h1>
<p>Temukan kelompokmu, bandingkan dengan {len(DF)} mahasiswa lain, dan simulasikan langkah menuju kebiasaan yang lebih baik.</p></div>{HERO_SVG}</div>""",
                unsafe_allow_html=True)
    info = [("👥", len(DF), "Total responden"), ("🎓", f"{DF.ipk.mean():.2f}", "Rata-rata IPK"),
            ("💸", rp(DF.pengeluaran.mean() * 1000), "Rata-rata pengeluaran/minggu"), ("🧩", K, "Cluster (K-Means)")]
    for kol, (i, v, t) in zip(st.columns(4), info):
        with kol:
            card(f'<div class="stat"><div class="ic">{i}</div><b>{v}</b><span>{t}</span></div>')
    best, boros, besar = int(np.argmax(M["ipk"])), int(np.argmax(M["peng"])), int(np.argmax(M["n"]))
    ins = [("🏆", "IPK tertinggi", f"{NAMA[best]} ({M['ipk'][best]})"),
           ("🔥", "Pengeluaran terbesar", f"{NAMA[boros]} ({rp(M['peng'][boros])}/mgg)"),
           ("👪", "Kelompok terbesar", f"{NAMA[besar]} ({M['n'][besar]} orang)")]
    for kol, (i, t, v) in zip(st.columns(3), ins):
        with kol:
            card(f'<div class="ins"><div class="ic" style="width:38px;height:38px;border-radius:12px;background:#E4ECFD;display:grid;place-items:center">{i}</div><div><span class="mut">{t}</span><br><b>{html.escape(v)}</b></div></div>')
    a, b = st.columns([1.7, 1])
    with a:
        card("<h4>Sebaran Cluster: Pengeluaran (X) vs IPK (Y)</h4><span class='mut'>Arahkan kursor ke titik untuk detail. ◆ = pusat cluster.</span>")
        peta(DF)
    with b:
        card("<h4>Proporsi Cluster</h4>")
        pie = pd.DataFrame({"nama": [NAMA[i] for i in range(K)], "n": M["n"]})
        st.altair_chart(alt.Chart(pie).mark_arc(innerRadius=62).encode(
            theta="n:Q", color=alt.Color("nama:N", scale=alt.Scale(domain=list(pie.nama), range=COL),
                                         legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["nama", "n"]).properties(height=330).configure(background="transparent"),
            width="stretch")
    card("<h4>Rata-rata Jam Aktivitas per Minggu</h4>")
    long = pd.DataFrame([(AKT[j], NAMA[i], M["akt"][i][j]) for i in range(K) for j in range(5)],
                        columns=["aktivitas", "nama", "jam"])
    st.altair_chart(alt.Chart(long).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("aktivitas:N", title=None, sort=AKT), xOffset="nama:N", y=alt.Y("jam:Q", title="Jam/minggu"),
        color=alt.Color("nama:N", scale=alt.Scale(domain=[NAMA[i] for i in range(K)], range=COL),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["nama", "aktivitas", "jam"]).properties(height=260).configure(background="transparent"),
        width="stretch")

# ============================== CEK CLUSTER ==================================
elif hal == "cek":
    judul("Cek Cluster Saya", "Masukkan data Anda, sistem menentukan kelompok yang paling sesuai")
    st.caption("Mulai cepat dengan contoh:")
    pc = st.columns([1, 1, 1, 4])
    for kol, n in zip(pc, PRESET):
        kol.button(f"{'🌱' if n == 'Hemat' else '⚖️' if n == 'Sedang' else '🔥'} {n}", on_click=isi, args=(n,), key="p" + n)
    kiri, kanan = st.columns(2)
    with kiri:
        with st.form("form"):
            ipk = st.number_input("IPK terakhir", 0.0, 4.0, step=0.01, key="ipk", help="Skala 0–4")
            st.markdown("**Pengeluaran per hari (Rp)**")
            kk = st.columns(4) + st.columns(3)
            peng = [kk[i].number_input(HARI[i], 0, 5_000_000, step=5000, key=f"d{i}") for i in range(7)]
            st.markdown("**Aktivitas per minggu (jam)** · opsional")
            ak = st.columns(5)
            jam = [ak[i].number_input(AKT[i], 0.0, 100.0, step=0.5, key=f"a{i}") for i in range(5)]
            kirim = st.form_submit_button("🔍 Kategorikan")
        if kirim:
            total = float(sum(peng))
            if total == 0:
                st.warning("Semua pengeluaran bernilai 0. Pastikan Anda mengisi pengeluaran yang sebenarnya.")
            if ipk == 0:
                st.warning("IPK bernilai 0. Periksa kembali isian Anda.")
            c, share = klasifikasi(total, ipk)
            st.session_state["res"] = dict(total=total, ipk=ipk, c=c, share=share, jam=jam)
            st.session_state["hist"].insert(0, {"IPK": ipk, "Pengeluaran/minggu": total, "Cluster": c, "Nama": NAMA[c]})
            st.toast(f"Anda masuk Cluster {c}: {NAMA[c]}", icon="🎉")
    R = st.session_state["res"]
    with kanan:
        if not R:
            card("<h4>Hasil</h4><p class='mut'>Isi data lalu tekan “Kategorikan”, atau pilih contoh di atas.</p>")
        else:
            c = R["c"]
            kt, kw = keyakinan(R["share"])
            bar = "".join(f'<div class="row"><span>{html.escape(NAMA[i])}</span><span>{R["share"][i]*100:.0f}%</span></div>'
                          f'<div class="bar"><i style="width:{R["share"][i]*100:.0f}%;background:{COL[i]}"></i></div>' for i in range(K))
            akt = "".join(f'<div class="row"><span>{AKT[j]}</span><span>{R["jam"][j]:g} jam · rata-rata {M["akt"][c][j]}</span></div>'
                          f'<div class="bar"><i style="width:{min(100, R["jam"][j]/max(R["jam"][j], M["akt"][c][j], 1)*100):.0f}%;background:{COL[c]}"></i></div>'
                          for j in range(5))
            cap = f" (dibatasi {rp(M['cap'])})" if R["total"] > M["cap"] else ""
            card(f"""<h4>Anda masuk…</h4>{pill(c)} <span class='chip' style='color:{kw}'>Keyakinan: {kt}</span>
<h2 style='margin:8px 0 2px'>{html.escape(NAMA[c])}</h2><p class='mut'>{rp(R['total'])}/minggu{cap} · IPK {R['ipk']}</p><p>{SARAN.get(c, '')}</p>
<div class='row'><span>Pengeluaran lebih tinggi dari</span><b>{persentil(R['total']/1000, 'pengeluaran')}% responden</b></div>
<div class='row'><span>IPK lebih tinggi dari</span><b>{persentil(R['ipk'], 'ipk')}% responden</b></div>
<h4 style='margin-top:14px'>Kedekatan ke tiap cluster</h4>{bar}
<h4 style='margin-top:10px'>Aktivitas Anda vs rata-rata cluster</h4>{akt}""", f'style="border-left:5px solid {COL[c]}"')
            st.info("Ingin tahu cara pindah cluster? Buka menu **🎯 Simulasi What-If**.")
    a, b = st.columns([1.7, 1])
    with a:
        card("<h4>Posisi Anda pada Peta Cluster</h4><span class='mut'>▲ merah = Anda</span>")
        peta(DF, (R["total"], R["ipk"]) if R else None)
    with b:
        card("<h4>Riwayat Pengecekan (sesi ini)</h4>")
        if st.session_state["hist"]:
            h = pd.DataFrame(st.session_state["hist"][:8])
            st.dataframe(h, hide_index=True, width="stretch")
            st.download_button("⬇ Unduh riwayat (CSV)", h.to_csv(index=False).encode(), "riwayat_cluster.csv", "text/csv")
        else:
            st.caption("Belum ada riwayat.")

# ============================ SIMULASI WHAT-IF ===============================
elif hal == "sim":
    judul("Simulasi What-If", "Coba ubah IPK atau pengeluaran, lihat apakah Anda pindah cluster")
    R = st.session_state["res"]
    b_tot, b_ipk = (R["total"], R["ipk"]) if R else (245000.0, 3.4)
    if not R:
        st.info("Belum ada data dari menu Cek Cluster. Simulasi memakai contoh: IPK 3,40 dan Rp245.000/minggu.")
    kiri, kanan = st.columns([1, 1.6])
    with kiri:
        card("<h4>Atur Skenario</h4>")
        s_ipk = st.slider("IPK simulasi", 0.0, 4.0, float(b_ipk), 0.01)
        s_pct = st.slider("Perubahan pengeluaran (%)", -70, 70, 0, 5)
        s_tot = b_tot * (1 + s_pct / 100)
        c0, _ = klasifikasi(b_tot, b_ipk)
        c1, sh = klasifikasi(s_tot, s_ipk)
        card(f"""<div class='row'><span>Sekarang</span><span>{pill(c0)}</span></div><div class='row' style='margin-top:6px'><span>Skenario</span><span>{pill(c1)}</span></div>
<p style='margin:10px 0 0'><b>{html.escape(NAMA[c1])}</b><br><span class='mut'>{rp(s_tot)}/minggu · IPK {s_ipk:.2f}</span></p>""")
        if c0 != c1:
            st.success(f"Skenario ini memindahkan Anda dari Cluster {c0} ke Cluster {c1}.")
        else:
            st.info("Skenario ini belum mengubah cluster Anda.")
    with kanan:
        card("<h4>Jalur pada Peta</h4><span class='mut'>▲ posisi sekarang · ＋ hijau = skenario</span>")
        peta(DF, (b_tot, b_ipk), 340, sim=((b_tot, b_ipk), (s_tot, s_ipk)))
    card("<h4>Cari Jalur ke Cluster Tujuan</h4>")
    tj = st.selectbox("Cluster tujuan", range(K), format_func=lambda i: f"Cluster {i} · {NAMA[i]}")
    o_ipk, o_p = butuh(tj, b_tot, b_ipk)
    if klasifikasi(b_tot, b_ipk)[0] == tj:
        st.success("Anda sudah berada di cluster ini.")
    else:
        x1, x2 = st.columns(2)
        x1.metric("Opsi A · Naikkan IPK ke", f"≥ {o_ipk:.2f}" if o_ipk else "Tidak cukup", f"+{o_ipk - b_ipk:.2f}" if o_ipk else None)
        x2.metric("Opsi B · Turunkan pengeluaran ke", f"≤ {rp(o_p)}" if o_p is not None else "Tidak cukup",
                  f"-{rp(b_tot - o_p)}" if o_p is not None else None, delta_color="inverse")
        st.caption("Masing-masing opsi dihitung dengan mengubah satu variabel saja, variabel lain tetap.")

# ============================ PROFIL & BANDINGKAN ============================
elif hal == "profil":
    judul("Profil & Bandingkan Cluster", "Pahami karakteristik tiap kelompok mahasiswa")
    t1, t2 = st.tabs(["Profil Tiap Cluster", "Bandingkan Dua Cluster"])
    with t1:
        for kol, c in zip(st.columns(K), range(K)):
            with kol:
                kp = M["komp"][c]
                seg = "".join(f'<i style="width:{v*100}%;background:hsl({215+i*18},70%,{40+i*6}%)" title="{KAT[i]}"></i>' for i, v in enumerate(kp))
                card(f"""{pill(c)}<h2 style='margin:8px 0 2px'>{html.escape(NAMA[c])}</h2>
<p class='mut'>{M['n'][c]} mahasiswa ({M['n'][c]/len(DF)*100:.0f}%)</p>
<div class='row'><span>Rata-rata IPK</span><b>{M['ipk'][c]}</b></div><div class='row'><span>Pengeluaran/minggu</span><b>{rp(M['peng'][c])}</b></div>
<div class='row'><span>Belajar · Organisasi · Kerja</span><b>{M['akt'][c][1]} · {M['akt'][c][2]} · {M['akt'][c][3]} jam</b></div>
<h4 style='margin-top:14px'>Komposisi pengeluaran</h4><div class='stack'>{seg}</div>
<p class='mut'>Terbesar: {KAT[int(np.argmax(kp))]} ({max(kp)*100:.0f}%)</p><p>{SARAN.get(c, '')}</p>""", f'style="border-top:5px solid {COL[c]}"')
    with t2:
        p, q = st.columns(2)
        ca = p.selectbox("Cluster A", range(K), 0, format_func=lambda i: NAMA[i])
        cb = q.selectbox("Cluster B", range(K), min(1, K - 1), format_func=lambda i: NAMA[i])
        baris = [("Jumlah mahasiswa", M["n"][ca], M["n"][cb]), ("Rata-rata IPK", M["ipk"][ca], M["ipk"][cb]),
                 ("Pengeluaran/minggu (Rp)", M["peng"][ca], M["peng"][cb])] + \
                [(f"Jam {AKT[j].lower()}", M["akt"][ca][j], M["akt"][cb][j]) for j in range(5)] + \
                [(f"% {KAT[j]}", round(M["komp"][ca][j] * 100, 1), round(M["komp"][cb][j] * 100, 1)) for j in range(8)]
        T = pd.DataFrame(baris, columns=["Metrik", NAMA[ca], NAMA[cb]])
        T["Selisih (B − A)"] = (T.iloc[:, 2] - T.iloc[:, 1]).round(2)
        st.dataframe(T, hide_index=True, width="stretch", height=470)

# ============================= EKSPLORASI DATA ===============================
elif hal == "eks":
    judul("Eksplorasi Data", "Filter data responden dan lihat sebarannya")
    f1, f2, f3 = st.columns(3)
    pilih = f1.multiselect("Cluster", list(range(K)), default=list(range(K)), format_func=lambda i: NAMA[i])
    r_ipk = f2.slider("Rentang IPK", 2.5, 4.0, (2.5, 4.0), 0.01)
    mx = int(DF.pengeluaran.max())
    r_p = f3.slider("Pengeluaran/minggu (ribu Rp)", 0, mx, (0, mx), 10)
    S = DF[DF.cluster.isin(pilih) & DF.ipk.between(*r_ipk) & DF.pengeluaran.between(*r_p)]
    card(f"<h4>{len(S)} dari {len(DF)} responden terpilih</h4>")
    if len(S):
        ta, tb, tc = st.tabs(["Peta", "Distribusi", "Tabel"])
        with ta:
            peta(S)
        with tb:
            sk = alt.Scale(domain=[NAMA[i] for i in range(K)], range=COL)
            h1 = alt.Chart(S).mark_bar(opacity=.85).encode(x=alt.X("ipk:Q", bin=alt.Bin(maxbins=18), title="IPK"), y="count()",
                                                           color=alt.Color("nama:N", scale=sk, legend=None)).properties(height=250)
            h2 = alt.Chart(S).mark_bar(opacity=.85).encode(x=alt.X("pengeluaran:Q", bin=alt.Bin(maxbins=18), title="Pengeluaran (ribu Rp)"),
                                                           y="count()", color=alt.Color("nama:N", scale=sk, legend=None)).properties(height=250)
            st.altair_chart((h1 | h2).configure(background="transparent"), width="stretch")
        with tc:
            st.dataframe(S.rename(columns={"pengeluaran": "pengeluaran (ribu Rp)"}), hide_index=True, width="stretch")
            st.download_button("⬇ Unduh data terfilter (CSV)", S.to_csv(index=False).encode(), "data_terfilter.csv", "text/csv")
    else:
        st.warning("Tidak ada data yang cocok dengan filter. Longgarkan rentang filter.")

# =============================== TENTANG METODE ==============================
else:
    import datetime as dt
    judul("Tentang Metode", "Bagaimana sistem ini bekerja, seberapa baik hasilnya, dan jadwal pelaksanaannya")
    U = METRIK["Model Utama (Rp × IPK)"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Algoritma", "K-Means")
    m2.metric("Jumlah cluster", K)
    m3.metric("Silhouette", f"{U['sil']:.4f}", help="Makin tinggi makin baik (>0,25 sedang, >0,5 kuat)")
    m4.metric("Davies-Bouldin", f"{U['db']:.4f}", help="Makin rendah makin baik")
    m5.metric("Calinski-Harabasz", f"{U['ch']:.2f}", help="Makin tinggi makin baik")
    st.caption("Angka di atas adalah hasil asli dari Colab (Bab V Evaluasi).")
    t1, t2, t3 = st.tabs(["Tahapan CRISP-DM", "Perbandingan Model", "Jadwal Pelaksanaan"])
    with t1:
        langkah = [("Business Understanding", "Memetakan kelompok mahasiswa berdasarkan pengeluaran dan prestasi akademik."),
                   ("Data Understanding", f"Data primer Google Form: {len(DF)} responden, pengeluaran 7 hari, IPK, dan jam aktivitas."),
                   ("Preprocessing", "Anonimisasi, validasi, mengeluarkan 1 responden yang mengisi hari belum terjadi, parsing rincian teks, penanganan outlier (IQR), standardisasi z-score."),
                   ("Model", "K-Means pada dua fitur: total pengeluaran mingguan dan IPK."),
                   ("Evaluasi", "Silhouette, Davies-Bouldin, Calinski-Harabasz, stabilitas bootstrap, uji Kruskal-Wallis."),
                   ("Penerapan", "Website ini: data baru dikelompokkan ke centroid terdekat pada skala z-score.")]
        for i, (j, dsc) in enumerate(langkah, 1):
            card(f'<div class="step"><div class="no">{i}</div><div><b>{j}</b><br><span class="mut">{dsc}</span></div></div>')
        st.caption(f"Batas pengeluaran (outlier, Q3 + 1,5·IQR): {rp(M['cap'])}/minggu. Nilai di atasnya dibatasi sebelum diklasifikasikan.")
    with t2:
        tb = pd.DataFrame(METRIK).T.rename(columns={"wcss": "Inertia (WCSS)", "sil": "Silhouette", "db": "Davies-Bouldin", "ch": "Calinski-Harabasz"})
        st.dataframe(tb, width="stretch")
        st.info("Model utama (Pengeluaran × IPK) memiliki silhouette jauh lebih baik daripada model 7 fitur, "
                "sehingga dipilih sebagai model penerapan. Jam aktivitas dipakai untuk memperkaya profil cluster.")
    with t3:
        s0, s1 = dt.date(2026, 9, 6), dt.date(2026, 11, 30)
        span = (s1 - s0).days
        hari = dt.date.today()
        pos = min(max((hari - s0).days / span * 100, 0), 100)
        for keg, pic, m, a in JADWAL:
            m, a = dt.date.fromisoformat(m), dt.date.fromisoformat(a)
            status, wr = ("Selesai", "#16A34A") if hari > a else ("Berjalan", "#F2994A") if hari >= m else ("Mendatang", "#6B7A99")
            kiri = (m - s0).days / span * 100
            lebar = ((a - m).days + 1) / span * 100
            card(f"""<div class='row'><b>{html.escape(keg)}</b><span class='chip' style='color:{wr}'>{status}</span></div>
<div style='position:relative;height:14px;background:#E4ECFD;border-radius:9px;margin:8px 0 4px'>
<i style='position:absolute;left:{kiri:.1f}%;width:{lebar:.1f}%;top:0;bottom:0;border-radius:9px;background:linear-gradient(90deg,#5B8DEF,#9B7BEA)'></i>
<i style='position:absolute;left:{pos:.1f}%;top:-3px;bottom:-3px;width:2px;background:#E5484D'></i></div>
<div class='row'><span class='mut'>{m.strftime('%d %b')} – {a.strftime('%d %b %Y')}</span><span class='mut'>PIC: {html.escape(pic)}</span></div>""")
        st.caption("Garis merah = hari ini. Rentang waktu: 6 September – 30 November 2026.")
