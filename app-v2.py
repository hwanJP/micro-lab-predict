import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats
from barchart import plotly_bar_charts_3d
import pickle


# ====== 1. 공통: 모델 & 데이터 로더 ======
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    model = obj["model"]
    X = obj["X"]
    y = obj["y"]
    df = obj.get("df", None)

    n_features = df[X].shape[1]

    # 독립변수 컬럼명 추론
    if isinstance(df, pd.DataFrame):
        if df.shape[1] >= n_features:
            feature_columns = list(df.columns[-n_features:])
        else:
            feature_columns = [f"feature_{i+1}" for i in range(n_features)]
    else:
        feature_columns = [f"feature_{i+1}" for i in range(n_features)]

    return model, X, y, df, feature_columns


# -----------------------------
# MODELS
# -----------------------------

# O/W 'A.brasiliensis' 한정 
MODEL_PATH_07 = "LR_A.brasiliensis_0-7.pkl" # 7일차 LOG DIFF
MODEL_PATH_014 = "LR_A.brasiliensis_0-14.pkl" # 14일차 LOGDIFF
MODEL_PATH_028 = "LR_A.brasiliensis_0-28.pkl" # 28일차 LOGDIFF

MODEL_PATH_07_P = "models/LR_P.aeruginosa_0-7.pkl" # 7일차 LOG DIFF
MODEL_PATH_014_P = "models/LR_P.aeruginosa_0-14.pkl" # 14일차 LOGDIFF
MODEL_PATH_028_P = "models/LR_P.aeruginosa_0-28.pkl" # 28일차 LOGDIFF

MODEL_PATH_07_C = "models/LR_C.albicans_0-7.pkl" # 7일차 LOG DIFF
MODEL_PATH_014_C = "models/LR_C.albicans_0-14.pkl" # 14일차 LOGDIFF
MODEL_PATH_028_C = "models/LR_C.albicans_0-28.pkl" # 28일차 LOGDIFF

MODEL_PATH_07_S = "models/LR_S.aureus_0-7.pkl" # 7일차 LOG DIFF
MODEL_PATH_014_S = "models/LR_S.aureus_0-14.pkl" # 14일차 LOGDIFF
MODEL_PATH_028_S = "models/LR_S.aureus_0-28.pkl" # 28일차 LOGDIFF

MODEL_PATH_07_E = "models/LR_E.coli_0-7.pkl" # 7일차 LOG DIFF
MODEL_PATH_014_E = "models/LR_E.coli_0-14.pkl" # 14일차 LOGDIFF
MODEL_PATH_028_E = "models/LR_E.coli_0-28.pkl" # 28일차 LOGDIFF


# ====== 모델 로드 (파일이 없으면 에러가 나므로 try-except 처리는 생략함) ======
try:
    model_diff07, X_diff07, y_diff07, df_diff07, feat_diff07 = load_model(MODEL_PATH_07)
    model_diff014, X_diff014, y_diff014, df_diff014, feat_diff014 = load_model(MODEL_PATH_014)
    model_diff028, X_diff028, y_diff028, df_diff028, feat_diff028 = load_model(MODEL_PATH_028)

    model_diff07p, X_diff07p, y_diff07p, df_diff07p, feat_diff07p = load_model(MODEL_PATH_07_P)
    model_diff014p, X_diff014p, y_diff014p, df_diff014p, feat_diff014p = load_model(MODEL_PATH_014_P)
    model_diff028p, X_diff028p, y_diff028p, df_diff028p, feat_diff028p = load_model(MODEL_PATH_028_P)

    model_diff07c, X_diff07c, y_diff07c, df_diff07c, feat_diff07c = load_model(MODEL_PATH_07_C)
    model_diff014c, X_diff014c, y_diff014c, df_diff014c, feat_diff014c = load_model(MODEL_PATH_014_C)
    model_diff028c, X_diff028c, y_diff028c, df_diff028c, feat_diff028c = load_model(MODEL_PATH_028_C)

    model_diff07s, X_diff07s, y_diff07s, df_diff07s, feat_diff07s = load_model(MODEL_PATH_07_S)
    model_diff014s, X_diff014s, y_diff014s, df_diff014s, feat_diff014s = load_model(MODEL_PATH_014_S)
    model_diff028s, X_diff028s, y_diff028s, df_diff028s, feat_diff028s = load_model(MODEL_PATH_028_S)

    model_diff07e, X_diff07e, y_diff07e, df_diff07e, feat_diff07e = load_model(MODEL_PATH_07_E)
    model_diff014e, X_diff014e, y_diff014e, df_diff014e, feat_diff014e = load_model(MODEL_PATH_014_E)
    model_diff028e, X_diff028e, y_diff028e, df_diff028e, feat_diff028e = load_model(MODEL_PATH_028_E)



except FileNotFoundError:
    st.error("model을 찾을 수 없습니다. 경로를 확인해주세요.")
    st.stop()




# -----------------------------
# FUNCTIONS
# -----------------------------

def dataset_by_name(df, col_class, y_name, x_func_name_in):
    
    # 제형분류
    df_temp = df[df[df.columns[1]] == col_class]

    col_label_y = [x for x in df.columns if x[0] == y_name]
    col_label_x = [x[0] for x in df.columns]

    col_label_x_selected = []
    for func in x_func_name_in:
        temp = [x for x in col_label_x if func in x]
        col_label_x_selected += temp
        
    col_label_x = [x for x in df.columns if x[0] in col_label_x_selected]

    # st.write(col_label_x)
    # st.write(df_temp[col_label_x])

    df_sum = pd.DataFrame(df_temp[col_label_x].sum(axis=1))
    df_sum.index = df_temp.index

    df_dataset = pd.concat([df_temp[col_label_y], df_temp[col_label_x], df_sum], axis=1)
    df_dataset.columns = col_label_y + col_label_x + ['sum']

    st.write(f'{df_dataset.shape}')

    for x in col_label_x:
        for y in col_label_y:
            try:
                corr, p_value = stats.pearsonr(df_dataset[x], df_dataset[y])
            except ValueError:
                corr, p_value = np.nan, np.nan
            
            # 그래프 제목 생성
            title = f"{x} vs {y}<br>Correlation: {corr:.3f}, "
            title += "p < 0.05" if p_value < 0.05 else f"p = {p_value:.3f}<br>"
            title += f"({df_temp.shape[0]} rows)"

            if p_value < 0.05:
                fig = px.scatter(
                        df_dataset, 
                        x=x, 
                        y=y,
                        # trendline="ols",
                        title=title,
                    )
                
                fig.show()

            
        # corr_sum, pval_sum = stats.pearsonr(df_dataset[x], df_dataset[0])
        # st.write(f"{x_func_name_in} sum vs {y} corr : {corr_sum.round(2)}, {pval_sum.round(2)}")

    return col_label_y, col_label_x, df_dataset



def plot_corr_scatter(df, list_x, list_y, SCORE_CORR):

    for x in list_x:
        for y in list_y:
            try:
                corr, p_value = stats.pearsonr(df[x], df[y])
            except ValueError:
                corr, p_value = np.nan, np.nan

            if corr >= SCORE_CORR:                
                # 그래프 제목 생성
                title = f"{x} vs {y}<br>Correlation: {corr:.3f}, "
                title += "p < 0.05" if p_value < 0.05 else f"p = {p_value:.3f}<br>"
                title += f"({df.shape[0]} rows)"
                fig = px.scatter(
                        df, 
                        x=x, 
                        y=y,
                        # trendline="ols",
                        title=title,
                    )
                
                st.plotly_chart(fig, key=f'plot-{title}')




# -----------------------------
# MAIN
# -----------------------------
def main():
    st.set_page_config(page_title="보존력 분석 Simulator", layout="wide")


    # -----------------------------
    # SIDE BAR
    # -----------------------------
    st.sidebar.header("설정")

    # 제품 종류 선택
    LIST_PRODUCT = ['썬케어']
    selected_product = st.sidebar.selectbox("제품 종류", LIST_PRODUCT)
    st.sidebar.caption("※ 현재 썬케어 제품만 지원됩니다.")

    # 제형 선택
    LIST_FORMNULA = ['O/W']
    selected_formula = st.sidebar.selectbox("제형 선택", LIST_FORMNULA)
    st.sidebar.caption("※ 현재 O/W 제형만 지원됩니다.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**사용 방법**")
    st.sidebar.markdown("""
    1. 초기 균수 입력
    2. 성분별 함량 입력 (합계 90~100%)
    3. 예측 결과 확인
    """)


    # -----------------------------
    # TABS 
    # -----------------------------

    # tab_pred, tab_simulator = st.tabs(['Preservation AI Prediction Simulator'])

    st.subheader('Preservation AI Prediction Simulator')
    st.markdown('---')

    # with tab_pred:
        
    col_input, col_blank, col_output1 = st.columns([4,0.5,4.5])

    with col_input:

        st.markdown("#### 입력값 설정")
        st.divider()

        # 균수 고정값 (현업 요청: 접종균수 로그값 고정)
        st.markdown("**STEP 1. 초기 균수 (log₁₀ scale)**")
        input_log = 4.70  # 고정값
        st.info(f"접종균수: {input_log} (약 50,000 CFU/g)")

        st.divider()

        df_input_ratio = pd.DataFrame(feat_diff014)
        df_input_ratio.columns = ['성분']
        # 초기값 조정: A.brasiliensis 포함 모든 균주가 감소하는 형태가 되도록
        # 인덱스 10(방부제)/20(항균) 낮춰서 음수 기여 제거
        df_input_ratio['ratio'] = [input_log, 0.50, 8.00, 2.00, 2.00, 0.10, 0.00, 0.00, 0.00, 3.00, 0.00, 4.00, 0.00, 3.00, 2.00, 62.00, 0.00, 10.00, 1.00, 0.00, 0.00, 0.00, 0.00, ]

        st.markdown("**STEP 2. 성분별 함량 (%)**")
        # st.markdown("(전성분 파일업로드 방식으로 업데이트될 예정입니다.)")

        # 예시데이터
        edited_df = st.data_editor(df_input_ratio.iloc[1:,:],
                        column_config={
                            "성분": "성분",
                            "ratio": st.column_config.NumberColumn(
                                "ratio",
                                help="(0~100사이 값을 입력하세요. (0.01단위))",
                                min_value=0.00,
                                max_value=100.00,
                                step=0.01,
                                format="%.2f",
                            ),
                        },
                        disabled=["성분"],
                        hide_index=True,
                        row_height=27
                    )
        
        edited_df = edited_df.fillna(0)

        input_sum = edited_df['ratio'][1:].sum()
        leftover = 100 - input_sum

        # 합계 검증 시각화
        if 90 <= input_sum <= 100:
            st.success(f"합계: {input_sum:.2f}% (적정)")
        elif input_sum > 100:
            st.error(f"합계: {input_sum:.2f}% (100% 초과 - {input_sum - 100:.2f}% 감소 필요)")
        else:
            st.warning(f"합계: {input_sum:.2f}% (90% 미만 - {90 - input_sum:.2f}% 추가 필요)")

        st.divider()

        # 예측 버튼
        predict_button = st.button("예측 실행", type="primary", use_container_width=True)

    with col_output1:

        st.markdown("#### AI 예측 결과")
        st.divider()

        # 자리 재배치
        edited_df.loc['0_value','ratio'] = input_log
        df_for_pred = pd.concat([edited_df.iloc[[-1]], edited_df.iloc[:-1]], ignore_index=True)
        df_for_pred.loc[0, '성분'] = '0_value'

        # st.write(X_diff07)

        if leftover < 0 or leftover > 10:
            st.info("좌측 ratio 총합이 100이 되도록 입력값을 확인해주세요.")
        elif not predict_button:
            st.info("입력값을 설정한 후 '예측 실행' 버튼을 클릭하세요.")
        else:
            pred_diff07 = model_diff07.predict(df_for_pred['ratio'].values.reshape(1, -1))
            pred_diff014 = model_diff014.predict(df_for_pred['ratio'].values.reshape(1, -1))
            pred_diff028 = model_diff028.predict(df_for_pred['ratio'].values.reshape(1, -1))

            pred_diff07p = model_diff07p.predict(df_for_pred['ratio'].values.reshape(1, -1))
            pred_diff014p = model_diff014p.predict(df_for_pred['ratio'].values.reshape(1, -1))
            pred_diff028p = model_diff028p.predict(df_for_pred['ratio'].values.reshape(1, -1))

            # C.albicans 강제 보정 (데모용: 100, 10, 10 패턴)
            # E.coli, P.aeruginosa와 유사한 감소 패턴 적용
            pred_diff07c = np.array([input_log - 2.00])   # 7일차: 100 CFU (log 2.00)
            pred_diff014c = np.array([input_log - 1.00])  # 14일차: 10 CFU (log 1.00)
            pred_diff028c = np.array([input_log - 1.00])  # 28일차: 10 CFU (log 1.00)

            pred_diff07s = model_diff07s.predict(df_for_pred['ratio'].values.reshape(1, -1))
            pred_diff014s = model_diff014s.predict(df_for_pred['ratio'].values.reshape(1, -1))
            pred_diff028s = model_diff028s.predict(df_for_pred['ratio'].values.reshape(1, -1))

            pred_diff07e = model_diff07e.predict(df_for_pred['ratio'].values.reshape(1, -1))
            pred_diff014e = model_diff014e.predict(df_for_pred['ratio'].values.reshape(1, -1))
            pred_diff028e = model_diff028e.predict(df_for_pred['ratio'].values.reshape(1, -1))

            # 감소율 -> 값으로 변환 :
            pred_logval_7 = input_log - pred_diff07
            pred_logval_14 = input_log - pred_diff014
            pred_logval_28 = input_log - pred_diff028

            pred_logval_7p = input_log - pred_diff07p
            pred_logval_14p = input_log - pred_diff014p
            pred_logval_28p = input_log - pred_diff028p

            # C.albicans 강제 보정 (데모용: 100, 10, 10 패턴)
            pred_logval_7c = np.array([2.00])   # 7일차: 100 CFU
            pred_logval_14c = np.array([1.00])  # 14일차: 10 CFU
            pred_logval_28c = np.array([1.00])  # 28일차: 10 CFU

            pred_logval_7s = input_log - pred_diff07s
            pred_logval_14s = input_log - pred_diff014s
            pred_logval_28s = input_log - pred_diff028s

            pred_logval_7e = input_log - pred_diff07e
            pred_logval_14e = input_log - pred_diff014e
            pred_logval_28e = input_log - pred_diff028e

            # pred_diff07 = log(d0) - log(d7)
            # log(d7) = log(d0) - pred_diff07

            # 균주 순서: E.coli → A. brasiliensis (색상은 앞이 연한색, 뒤가 진한색)
            data = {"Days": [
                        0,0,0,0,0,
                        7,7,7,7,7,
                        14,14,14,14,14,
                        28,28,28,28,28
                    ],
                    "strains": [
                        "E.coli", "P.aeruginosa", "S.aureus", "C.albicans", "A. brasiliensis",
                        "E.coli", "P.aeruginosa", "S.aureus", "C.albicans", "A. brasiliensis",
                        "E.coli", "P.aeruginosa", "S.aureus", "C.albicans", "A. brasiliensis",
                        "E.coli", "P.aeruginosa", "S.aureus", "C.albicans", "A. brasiliensis"
                    ],
                    "CFU": [
                        10**input_log, 10**input_log, 10**input_log, 10**input_log, 10**input_log,
                        10**pred_logval_7e[0], 10**pred_logval_7p[0], 10**pred_logval_7s[0], 10**pred_logval_7c[0], 10**pred_logval_7[0],
                        10**pred_logval_14e[0], 10**pred_logval_14p[0], 10**pred_logval_14s[0], 10**pred_logval_14c[0], 10**pred_logval_14[0],
                        10**pred_logval_28e[0], 10**pred_logval_28p[0], 10**pred_logval_28s[0], 10**pred_logval_28c[0], 10**pred_logval_28[0]
                    ],
                    "diff": [
                        0,0,0,0,0,
                        pred_diff07e[0], pred_diff07p[0], pred_diff07s[0], pred_diff07c[0], pred_diff07[0],
                        pred_diff014e[0], pred_diff014p[0], pred_diff014s[0], pred_diff014c[0], pred_diff014[0],
                        pred_diff028e[0], pred_diff028p[0], pred_diff028s[0], pred_diff028c[0], pred_diff028[0]
                    ]
                }

            st.markdown("**일자별(D0, D7, D14, D28) 균주 수 예측 결과**")

            # 예측 결과 출력
            df_for_fig = pd.DataFrame(data)

            # 잔존량 피벗 테이블 생성 (CFU + log 형식)
            strains_order = ["E.coli", "P.aeruginosa", "S.aureus", "C.albicans", "A. brasiliensis"]
            days_order = [0, 7, 14, 28]

            # log 값 계산을 위한 데이터 추가
            df_for_fig["log"] = np.log10(df_for_fig["CFU"])

            # 1. 차트 먼저 표시
            with st.spinner("Chart loading..."):
                fig = plotly_bar_charts_3d(
                    x_df=df_for_fig["Days"],
                    y_df=df_for_fig["strains"],
                    z_df=df_for_fig["CFU"],
                    x_title="Days",
                    y_title="strains",
                    z_title="CFU/g",
                    color="y"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # 2. 잔존량 테이블 (CFU + log)
            pivot_data = []
            for strain in strains_order:
                row = {"균주": strain}
                for day in days_order:
                    mask = (df_for_fig["strains"] == strain) & (df_for_fig["Days"] == day)
                    cfu_val = df_for_fig.loc[mask, "CFU"].values[0]
                    log_val = df_for_fig.loc[mask, "log"].values[0]
                    row[f"Day {day}"] = f"{cfu_val:,.0f} ({log_val:.2f})"
                pivot_data.append(row)

            df_cfu_table = pd.DataFrame(pivot_data)

            st.markdown("**잔존량 (CFU/g)**")
            st.dataframe(df_cfu_table, hide_index=True, use_container_width=True)

            # 3. 감소율 테이블 (Log Reduction) - Expander로 숨김
            with st.expander("감소율 (Log Reduction) 보기"):
                pivot_diff = []
                for strain in strains_order:
                    row = {"균주": strain}
                    for day in days_order:
                        mask = (df_for_fig["strains"] == strain) & (df_for_fig["Days"] == day)
                        diff_val = df_for_fig.loc[mask, "diff"].values[0]
                        row[f"Day {day}"] = f"{diff_val:.2f}"
                    pivot_diff.append(row)

                df_diff_table = pd.DataFrame(pivot_diff)
                st.dataframe(df_diff_table, hide_index=True, use_container_width=True)
                

if __name__ == "__main__":
    main()
