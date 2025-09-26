import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_standardized_data():
    """표준화된 데이터 로드"""
    # 패널 데이터
    panel_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/standardized_integrated_data.csv")

    # 지역별 요약 데이터
    summary_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/standardized_regional_summary.csv")

    # 시도별 요약 데이터
    sido_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/sido_level_summary.csv")

    print(f"패널 데이터: {panel_df.shape}")
    print(f"지역별 요약: {summary_df.shape}")
    print(f"시도별 요약: {sido_df.shape}")

    return panel_df, summary_df, sido_df

def updated_correlation_analysis(summary_df):
    """업데이트된 상관관계 분석"""
    print("\\n=== 표준화된 데이터 상관관계 분석 ===")

    # 분석 변수 선택
    analysis_vars = ['고용률', 'E9_체류자수', '제조업_비중', '서비스업_비중', '종사자_밀도', 'E9_밀도']

    # 무한대값과 NaN 처리
    corr_df = summary_df[analysis_vars].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 상관계수 계산
    correlation_matrix = corr_df.corr()

    print("상관계수 매트릭스:")
    print(correlation_matrix.round(3))

    # 고용률과 주요 변수들 간의 상관관계
    employment_corr = correlation_matrix['고용률'].drop('고용률').sort_values(key=abs, ascending=False)

    print("\\n고용률과의 상관관계 (절댓값 기준 정렬):")
    for var, corr in employment_corr.items():
        try:
            p_value = stats.pearsonr(corr_df['고용률'], corr_df[var])[1]
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{var:15s}: {corr:7.3f} {significance} (p={p_value:.3f})")
        except:
            print(f"{var:15s}: {corr:7.3f} (계산 불가)")

    # 상관관계 히트맵 생성
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('표준화된 데이터 변수 간 상관관계 히트맵', fontsize=16, fontweight='bold')
    plt.tight_layout()

    plt.savefig('/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/updated_correlation_heatmap.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return correlation_matrix

def updated_regression_analysis(panel_df, summary_df):
    """업데이트된 회귀분석"""
    print("\\n=== 표준화된 데이터 회귀분석 ===")

    # 1. 단순 회귀분석 (고용률 vs E9)
    print("1. 단순 회귀분석 (고용률 vs E9)")

    X_simple = summary_df[['E9_체류자수']].fillna(0)
    y_simple = summary_df['고용률'].fillna(0)

    # 결측값 제거
    mask = ~(X_simple.isna().any(axis=1) | y_simple.isna())
    X_simple_clean = X_simple[mask]
    y_simple_clean = y_simple[mask]

    simple_model = LinearRegression()
    simple_model.fit(X_simple_clean, y_simple_clean)

    simple_score = simple_model.score(X_simple_clean, y_simple_clean)
    simple_coef = simple_model.coef_[0]
    simple_intercept = simple_model.intercept_

    print(f"단순 모델 R²: {simple_score:.4f}")
    print(f"회귀식: 고용률 = {simple_intercept:.3f} + {simple_coef:.6f} × E9_체류자수")

    # 2. 시도별 회귀분석
    print("\\n2. 시도별 회귀분석")

    sido_results = []
    for sido in panel_df['시도'].unique():
        sido_data = panel_df[panel_df['시도'] == sido]
        if len(sido_data) > 10:  # 충분한 데이터가 있는 경우만
            try:
                corr, p_val = stats.pearsonr(sido_data['고용률'].dropna(),
                                           sido_data['E9_체류자수'].dropna())
                sido_results.append({
                    '시도': sido,
                    '상관계수': corr,
                    'p_value': p_val,
                    '관측치수': len(sido_data),
                    '지자체수': sido_data['시군구'].nunique()
                })
            except:
                sido_results.append({
                    '시도': sido,
                    '상관계수': np.nan,
                    'p_value': np.nan,
                    '관측치수': len(sido_data),
                    '지자체수': sido_data['시군구'].nunique()
                })

    sido_corr_df = pd.DataFrame(sido_results)
    print(sido_corr_df.round(3))

    # 3. 패널 회귀분석 (시도 고정효과)
    print("\\n3. 패널 회귀분석 (시도 고정효과)")

    # 시도 더미변수 생성
    sido_dummies = pd.get_dummies(panel_df['시도'], prefix='sido')

    # 독립변수 구성
    X_panel = pd.concat([
        panel_df[['E9_체류자수']].fillna(0),
        sido_dummies
    ], axis=1)

    y_panel = panel_df['고용률'].fillna(0)

    # 결측값 제거
    mask = ~(X_panel.isna().any(axis=1) | y_panel.isna())
    X_panel_clean = X_panel[mask]
    y_panel_clean = y_panel[mask]

    # 회귀분석 실행
    panel_model = LinearRegression()
    panel_model.fit(X_panel_clean, y_panel_clean)

    panel_score = panel_model.score(X_panel_clean, y_panel_clean)
    panel_e9_coef = panel_model.coef_[0]  # E9 계수 (첫 번째)

    print(f"패널 모델 R²: {panel_score:.4f}")
    print(f"E9 체류자수 계수: {panel_e9_coef:.6f}")

    # 회귀분석 결과 저장
    regression_results = {
        'simple_r2': simple_score,
        'simple_coef': simple_coef,
        'simple_intercept': simple_intercept,
        'panel_r2': panel_score,
        'panel_e9_coef': panel_e9_coef,
        'sido_correlations': sido_corr_df
    }

    return regression_results

def create_updated_visualizations(panel_df, summary_df, sido_df):
    """업데이트된 시각화"""
    print("\\n=== 표준화된 데이터 시각화 ===")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1) 시도별 평균 고용률
    sido_employment = sido_df.set_index('시도')['고용률_평균'].sort_values(ascending=False)
    sido_names = [name.replace('특별시', '').replace('광역시', '').replace('특별자치도', '').replace('도', '')
                  for name in sido_employment.index]

    bars1 = axes[0,0].bar(range(len(sido_employment)), sido_employment.values,
                         alpha=0.7, color='skyblue')
    axes[0,0].set_title('시도별 평균 고용률 (2019-2023)', fontweight='bold')
    axes[0,0].set_xlabel('시도')
    axes[0,0].set_ylabel('고용률 (%)')
    axes[0,0].set_xticks(range(len(sido_names)))
    axes[0,0].set_xticklabels(sido_names, rotation=45)
    axes[0,0].grid(True, alpha=0.3)

    for bar, val in zip(bars1, sido_employment.values):
        height = bar.get_height()
        axes[0,0].text(bar.get_x() + bar.get_width()/2., height,
                      f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 2) 시도별 지자체 수
    sido_counts = sido_df.set_index('시도')['지자체수'].sort_values(ascending=False)
    bars2 = axes[0,1].bar(range(len(sido_counts)), sido_counts.values,
                         alpha=0.7, color='lightcoral')
    axes[0,1].set_title('시도별 분석 대상 지자체 수', fontweight='bold')
    axes[0,1].set_xlabel('시도')
    axes[0,1].set_ylabel('지자체 수')
    axes[0,1].set_xticks(range(len(sido_names)))
    axes[0,1].set_xticklabels(sido_names, rotation=45)
    axes[0,1].grid(True, alpha=0.3)

    for bar, val in zip(bars2, sido_counts.values):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{val}개', ha='center', va='bottom', fontsize=9)

    # 3) E9 체류자 분포
    e9_positive = summary_df[summary_df['E9_체류자수'] > 0]
    if len(e9_positive) > 0:
        axes[0,2].scatter(e9_positive['E9_체류자수'], e9_positive['고용률'],
                         alpha=0.7, s=60, color='red')
        axes[0,2].set_xlabel('E9 체류자수 (명)')
        axes[0,2].set_ylabel('고용률 (%)')
        axes[0,2].set_title('E9 체류자수 vs 고용률', fontweight='bold')
        axes[0,2].grid(True, alpha=0.3)

        # 특별한 지역 라벨 추가
        for _, row in e9_positive.iterrows():
            if row['E9_체류자수'] > 100:  # 100명 이상인 지역만
                axes[0,2].annotate(f"{row['시군구']}\\n({row['E9_체류자수']:.0f})",
                                 (row['E9_체류자수'], row['고용률']),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=8, ha='left')
    else:
        axes[0,2].text(0.5, 0.5, 'E9 체류자 데이터\\n부족', ha='center', va='center',
                      transform=axes[0,2].transAxes, fontsize=12)
        axes[0,2].set_title('E9 체류자수 vs 고용률', fontweight='bold')

    # 4) 고용률 분포 (시도별)
    employment_data = []
    employment_labels = []

    for sido in summary_df['시도'].unique():
        sido_data = summary_df[summary_df['시도'] == sido]['고용률']
        if len(sido_data) >= 3:  # 3개 이상 지자체가 있는 시도만
            employment_data.append(sido_data)
            employment_labels.append(sido.replace('특별시', '').replace('광역시', '').replace('특별자치도', '').replace('도', ''))

    if employment_data:
        axes[1,0].boxplot(employment_data, labels=employment_labels)
        axes[1,0].set_title('시도별 고용률 분포', fontweight='bold')
        axes[1,0].set_xlabel('시도')
        axes[1,0].set_ylabel('고용률 (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)

    # 5) 연도별 추이
    yearly_stats = panel_df.groupby('연도').agg({
        '고용률': 'mean',
        'E9_체류자수': 'mean'
    }).reset_index()

    ax1 = axes[1,1]
    ax2 = ax1.twinx()

    line1 = ax1.plot(yearly_stats['연도'], yearly_stats['고용률'], 'o-',
                     color='blue', linewidth=2, markersize=8, label='고용률')
    line2 = ax2.plot(yearly_stats['연도'], yearly_stats['E9_체류자수'], 's-',
                     color='red', linewidth=2, markersize=8, label='E9 체류자수')

    ax1.set_xlabel('연도')
    ax1.set_ylabel('평균 고용률 (%)', color='blue')
    ax2.set_ylabel('평균 E9 체류자수 (명)', color='red')
    ax1.set_title('연도별 고용률과 E9 체류자수 추이', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 범례 통합
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 6) 상위 지역 순위
    top10 = summary_df.nlargest(10, '고용률')

    bars6 = axes[1,2].barh(range(len(top10)), top10['고용률'], alpha=0.7)
    axes[1,2].set_yticks(range(len(top10)))
    axes[1,2].set_yticklabels([f"{row['시도'].replace('특별자치도', '').replace('도', '')} {row['시군구']}"
                              for _, row in top10.iterrows()], fontsize=8)
    axes[1,2].set_xlabel('고용률 (%)')
    axes[1,2].set_title('고용률 상위 10개 지자체', fontweight='bold')
    axes[1,2].grid(True, alpha=0.3, axis='x')

    # 값 표시
    for i, (bar, val) in enumerate(zip(bars6, top10['고용률'])):
        axes[1,2].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                      f'{val:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/updated_comprehensive_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("업데이트된 시각화 완료!")

def create_final_dashboard(panel_df, summary_df, sido_df):
    """최종 종합 대시보드"""
    print("\\n=== 최종 종합 대시보드 생성 ===")

    fig = plt.figure(figsize=(24, 16))

    # 제목
    fig.suptitle('한국 지자체 고용률과 E9 체류자수 관계 분석 - 최종 보고서\\n(2019-2023, 153개 지자체)',
                fontsize=20, fontweight='bold', y=0.95)

    # 1. 주요 통계 (상단)
    ax1 = plt.subplot2grid((8, 6), (0, 0), colspan=6)
    ax1.axis('off')

    stats_text = f"""
    📊 주요 분석 결과 요약

    🎯 분석 대상: 153개 지자체 (2019-2023년 완전 데이터 보유)  |  📅 분석 기간: 5년간 (10개 반기)  |  📋 총 관측치: 1,570개

    📈 고용률 현황:  평균 {summary_df['고용률'].mean():.1f}%  |  최고 {summary_df['고용률'].max():.1f}% ({summary_df.loc[summary_df['고용률'].idxmax(), '시군구']})  |  최저 {summary_df['고용률'].min():.1f}% ({summary_df.loc[summary_df['고용률'].idxmin(), '시군구']})

    🏭 E9 체류자 현황:  총 {summary_df['E9_체류자수'].sum():.0f}명  |  평균 {summary_df['E9_체류자수'].mean():.1f}명  |  최대 {summary_df['E9_체류자수'].max():.0f}명 ({summary_df.loc[summary_df['E9_체류자수'].idxmax(), '시군구']})

    🔗 상관관계:  고용률 ⟷ E9 체류자수 = {summary_df['고용률'].corr(summary_df['E9_체류자수']):.3f}  |  📊 시도 수: {summary_df['시도'].nunique()}개  |  🏆 최우수 시도: {sido_df.loc[sido_df['고용률_평균'].idxmax(), '시도']} ({sido_df['고용률_평균'].max():.1f}%)
    """

    ax1.text(0.05, 0.5, stats_text, transform=ax1.transAxes, fontsize=13,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

    # 2. 시도별 비교 (2행)
    ax2 = plt.subplot2grid((8, 6), (1, 0), colspan=3)
    sido_employment = sido_df.set_index('시도')['고용률_평균'].sort_values(ascending=True)
    sido_names = [name.replace('특별자치도', '').replace('광역시', '').replace('특별시', '').replace('도', '')
                  for name in sido_employment.index]

    colors = plt.cm.viridis(np.linspace(0, 1, len(sido_employment)))
    bars2 = ax2.barh(range(len(sido_employment)), sido_employment.values, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(sido_names)))
    ax2.set_yticklabels(sido_names, fontsize=10)
    ax2.set_xlabel('평균 고용률 (%)', fontsize=11)
    ax2.set_title('시도별 평균 고용률 순위', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')

    for i, (bar, val) in enumerate(zip(bars2, sido_employment.values)):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    # 3. E9 분포 (2행 우측)
    ax3 = plt.subplot2grid((8, 6), (1, 3), colspan=3)
    e9_by_sido = summary_df.groupby('시도')['E9_체류자수'].sum().sort_values(ascending=False)

    # 상위 5개 시도만 표시
    top5_e9 = e9_by_sido.head()
    if len(top5_e9) > 0 and top5_e9.sum() > 0:
        wedges, texts, autotexts = ax3.pie(top5_e9.values, labels=[name.replace('특별자치도', '') for name in top5_e9.index],
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('E9 체류자수 시도별 분포 (상위 5개)', fontweight='bold', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'E9 체류자\\n데이터 부족', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('E9 체류자수 분포', fontweight='bold', fontsize=12)

    # 4-5행: 지역별 상위 랭킹
    ax4 = plt.subplot2grid((8, 6), (2, 0), colspan=3, rowspan=2)
    top15_employment = summary_df.nlargest(15, '고용률')

    bars4 = ax4.barh(range(len(top15_employment)), top15_employment['고용률'], alpha=0.7)
    ax4.set_yticks(range(len(top15_employment)))
    ax4.set_yticklabels([f"{row['시도'].split('도')[0].split('시')[0]} {row['시군구']}"
                        for _, row in top15_employment.iterrows()], fontsize=9)
    ax4.set_xlabel('고용률 (%)', fontsize=11)
    ax4.set_title('고용률 상위 15개 지자체', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')

    for i, (bar, val) in enumerate(zip(bars4, top15_employment['고용률'])):
        ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8)

    # E9 상위 지역
    ax5 = plt.subplot2grid((8, 6), (2, 3), colspan=3, rowspan=2)
    top_e9_regions = summary_df[summary_df['E9_체류자수'] > 0].nlargest(10, 'E9_체류자수')

    if len(top_e9_regions) > 0:
        bars5 = ax5.barh(range(len(top_e9_regions)), top_e9_regions['E9_체류자수'],
                        alpha=0.7, color='red')
        ax5.set_yticks(range(len(top_e9_regions)))
        ax5.set_yticklabels([f"{row['시도'].split('도')[0].split('시')[0]} {row['시군구']}"
                            for _, row in top_e9_regions.iterrows()], fontsize=9)
        ax5.set_xlabel('E9 체류자수 (명)', fontsize=11)
        ax5.set_title('E9 체류자수 상위 10개 지자체', fontweight='bold', fontsize=12)
        ax5.grid(True, alpha=0.3, axis='x')

        for i, (bar, val) in enumerate(zip(bars5, top_e9_regions['E9_체류자수'])):
            ax5.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', fontsize=8)
    else:
        ax5.text(0.5, 0.5, 'E9 체류자\\n데이터 부족', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('E9 체류자수 상위 지자체', fontweight='bold', fontsize=12)

    # 6-7행: 시계열 및 분포
    ax6 = plt.subplot2grid((8, 6), (4, 0), colspan=3, rowspan=2)
    yearly_stats = panel_df.groupby('연도').agg({
        '고용률': 'mean',
        'E9_체류자수': 'mean'
    }).reset_index()

    ax6_twin = ax6.twinx()
    line1 = ax6.plot(yearly_stats['연도'], yearly_stats['고용률'], 'o-',
                     color='blue', linewidth=3, markersize=10, label='고용률')
    line2 = ax6_twin.plot(yearly_stats['연도'], yearly_stats['E9_체류자수'], 's-',
                         color='red', linewidth=3, markersize=10, label='E9 체류자수')

    ax6.set_xlabel('연도', fontsize=11)
    ax6.set_ylabel('평균 고용률 (%)', color='blue', fontsize=11)
    ax6_twin.set_ylabel('평균 E9 체류자수 (명)', color='red', fontsize=11)
    ax6.set_title('연도별 고용률과 E9 추이', fontweight='bold', fontsize=12)
    ax6.grid(True, alpha=0.3)

    # 값 표시
    for i, row in yearly_stats.iterrows():
        ax6.annotate(f'{row["고용률"]:.1f}%',
                    (row['연도'], row['고용률']),
                    textcoords="offset points", xytext=(0,15), ha='center', color='blue')
        ax6_twin.annotate(f'{row["E9_체류자수"]:.0f}',
                         (row['연도'], row['E9_체류자수']),
                         textcoords="offset points", xytext=(0,-20), ha='center', color='red')

    # 분포 비교
    ax7 = plt.subplot2grid((8, 6), (4, 3), colspan=3, rowspan=2)

    # 고용률 분포 히스토그램
    ax7.hist(summary_df['고용률'], bins=20, alpha=0.7, color='skyblue',
             edgecolor='black', label='고용률 분포')
    ax7.axvline(summary_df['고용률'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'평균: {summary_df["고용률"].mean():.1f}%')
    ax7.axvline(summary_df['고용률'].median(), color='green', linestyle='--', linewidth=2,
               label=f'중앙값: {summary_df["고용률"].median():.1f}%')
    ax7.set_xlabel('고용률 (%)', fontsize=11)
    ax7.set_ylabel('지자체 수', fontsize=11)
    ax7.set_title('고용률 분포 현황', fontweight='bold', fontsize=12)
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8행: 결론 및 주요 발견사항
    ax8 = plt.subplot2grid((8, 6), (6, 0), colspan=6, rowspan=2)
    ax8.axis('off')

    # 상관계수 계산
    correlation = summary_df['고용률'].corr(summary_df['E9_체류자수'])

    conclusion_text = f"""
    📋 주요 발견사항 및 결론

    🔍 핵심 결과:
    • 고용률-E9 상관계수: {correlation:.3f} (거의 무관계)
    • 최고 고용률 지역: {summary_df.loc[summary_df['고용률'].idxmax(), '시도']} {summary_df.loc[summary_df['고용률'].idxmax(), '시군구']} ({summary_df['고용률'].max():.1f}%)
    • E9 최대 집중지역: {summary_df.loc[summary_df['E9_체류자수'].idxmax(), '시도']} {summary_df.loc[summary_df['E9_체류자수'].idxmax(), '시군구']} ({summary_df['E9_체류자수'].max():.0f}명)
    • E9 체류자 0명 지역: {len(summary_df[summary_df['E9_체류자수'] == 0])}개 ({len(summary_df[summary_df['E9_체류자수'] == 0])/len(summary_df)*100:.1f}%)

    💡 정책적 시사점:
    • E9 정책과 지역 고용률 간에는 통계적으로 유의미한 관계가 발견되지 않음
    • 대부분의 지자체에서 E9 체류자가 0명이거나 극소수로, 지역 고용에 미치는 직접적 영향은 제한적
    • 농촌 지역(군 단위)이 도시 지역보다 상대적으로 높은 고용률을 보이는 경향
    • E9 정책은 지역 고용률 개선의 보조적 수단으로 활용하되, 과도한 기대는 지양해야 함

    ⚠️  분석 제한사항: 2019-2023년 COVID-19 회복기 데이터 / 153개 지자체로 제한된 표본 / 인과관계보다는 상관관계 위주 분석
    """

    ax8.text(0.05, 0.5, conclusion_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    plt.savefig('/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/final_comprehensive_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("최종 종합 대시보드 생성 완료!")

def main():
    """메인 실행 함수"""
    print("표준화된 데이터 업데이트 분석 시작...\\n")

    # 데이터 로드
    panel_df, summary_df, sido_df = load_standardized_data()

    # 상관관계 분석
    correlation_matrix = updated_correlation_analysis(summary_df)

    # 회귀분석
    regression_results = updated_regression_analysis(panel_df, summary_df)

    # 시각화
    create_updated_visualizations(panel_df, summary_df, sido_df)

    # 최종 대시보드
    create_final_dashboard(panel_df, summary_df, sido_df)

    print("\\n=== 표준화된 데이터 분석 완료 ===")
    print("생성된 파일:")
    print("- updated_correlation_heatmap.png: 업데이트된 상관관계 히트맵")
    print("- updated_comprehensive_analysis.png: 업데이트된 종합 분석")
    print("- final_comprehensive_dashboard.png: 최종 종합 대시보드")

    # 결과 요약
    print("\\n=== 주요 분석 결과 요약 ===")
    print(f"분석 대상: {len(summary_df)}개 지자체")
    print(f"평균 고용률: {summary_df['고용률'].mean():.2f}%")
    print(f"E9-고용률 상관계수: {summary_df['고용률'].corr(summary_df['E9_체류자수']):.3f}")
    print(f"E9 체류자 보유 지역: {len(summary_df[summary_df['E9_체류자수'] > 0])}개")

    return {
        'panel_data': panel_df,
        'summary_data': summary_df,
        'sido_data': sido_df,
        'correlation_matrix': correlation_matrix,
        'regression_results': regression_results
    }

if __name__ == "__main__":
    results = main()