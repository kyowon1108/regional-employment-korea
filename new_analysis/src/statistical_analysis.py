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

def load_data():
    """통합 데이터 로드"""
    df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/integrated_analysis_data.csv")

    print(f"데이터 크기: {df.shape}")
    print(f"지자체 수: {df[['시도', '시군구']].drop_duplicates().shape[0]}개")
    print(f"분석 기간: {df['연도'].min()}-{df['연도'].max()}년")

    return df

def calculate_averages(df):
    """5년 평균값 계산"""
    print("\\n=== 5년 평균값 계산 ===")

    # 지자체별 5년 평균 계산
    avg_df = df.groupby(['시도', '시군구']).agg({
        '고용률': 'mean',
        'E9_체류자수': 'mean',
        '제조업_종사자수': 'mean',
        '서비스업_종사자수': 'mean',
        '전체_종사자수': 'mean',
        '면적': 'first'
    }).reset_index()

    # 추가 지표 계산
    avg_df['제조업_비중'] = avg_df['제조업_종사자수'] / avg_df['전체_종사자수'] * 100
    avg_df['서비스업_비중'] = avg_df['서비스업_종사자수'] / avg_df['전체_종사자수'] * 100
    avg_df['종사자_밀도'] = avg_df['전체_종사자수'] / avg_df['면적']
    avg_df['E9_밀도'] = avg_df['E9_체류자수'] / avg_df['면적']

    # 무한대 및 NaN 값 처리
    avg_df = avg_df.replace([np.inf, -np.inf], np.nan)
    avg_df = avg_df.fillna(0)

    print(f"평균 데이터 크기: {avg_df.shape}")
    print("\\n기본 통계:")
    print(avg_df[['고용률', 'E9_체류자수', '제조업_비중', '서비스업_비중']].describe().round(2))

    return avg_df

def correlation_analysis(avg_df):
    """상관관계 분석"""
    print("\\n=== 상관관계 분석 ===")

    # 분석 변수 선택
    analysis_vars = ['고용률', 'E9_체류자수', '제조업_비중', '서비스업_비중', '종사자_밀도', 'E9_밀도']
    corr_df = avg_df[analysis_vars]

    # 상관계수 계산
    correlation_matrix = corr_df.corr()

    print("상관계수 매트릭스:")
    print(correlation_matrix.round(3))

    # 고용률과 주요 변수들 간의 상관관계
    employment_corr = correlation_matrix['고용률'].drop('고용률').sort_values(key=abs, ascending=False)

    print("\\n고용률과의 상관관계 (절댓값 기준 정렬):")
    for var, corr in employment_corr.items():
        p_value = stats.pearsonr(avg_df['고용률'], avg_df[var])[1]
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"{var:15s}: {corr:7.3f} {significance} (p={p_value:.3f})")

    # 상관관계 히트맵 생성
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('변수 간 상관관계 히트맵', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 저장
    plt.savefig('/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/correlation_heatmap.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return correlation_matrix

def regression_analysis(df, avg_df):
    """회귀분석"""
    print("\\n=== 회귀분석 ===")

    # 1. 패널 회귀분석 (시간 고정효과)
    print("1. 패널 회귀분석 (연도 고정효과)")

    # 더미변수 생성
    panel_df = df.copy()
    year_dummies = pd.get_dummies(panel_df['연도'], prefix='year')
    panel_df = pd.concat([panel_df, year_dummies], axis=1)

    # 독립변수와 종속변수 설정
    X_panel = panel_df[['E9_체류자수'] + list(year_dummies.columns)]
    y_panel = panel_df['고용률']

    # 결측값 제거
    mask = ~(X_panel.isna().any(axis=1) | y_panel.isna())
    X_panel_clean = X_panel[mask]
    y_panel_clean = y_panel[mask]

    # 회귀분석 실행
    panel_model = LinearRegression()
    panel_model.fit(X_panel_clean, y_panel_clean)

    panel_score = panel_model.score(X_panel_clean, y_panel_clean)
    panel_coef = panel_model.coef_[0]  # E9 계수

    print(f"패널 모델 R²: {panel_score:.4f}")
    print(f"E9 체류자수 계수: {panel_coef:.6f}")

    # 2. 횡단면 회귀분석 (5년 평균 기준)
    print("\\n2. 횡단면 회귀분석 (5년 평균)")

    # 독립변수와 종속변수 설정
    X_vars = ['E9_체류자수', '제조업_비중', '서비스업_비중', '종사자_밀도']
    X_cross = avg_df[X_vars]
    y_cross = avg_df['고용률']

    # 결측값과 무한대 제거
    mask = ~(X_cross.isna().any(axis=1) | y_cross.isna() |
             np.isinf(X_cross).any(axis=1) | np.isinf(y_cross))
    X_cross_clean = X_cross[mask]
    y_cross_clean = y_cross[mask]

    # 표준화
    scaler = StandardScaler()
    X_cross_scaled = scaler.fit_transform(X_cross_clean)

    # 회귀분석 실행
    cross_model = LinearRegression()
    cross_model.fit(X_cross_scaled, y_cross_clean)

    cross_score = cross_model.score(X_cross_scaled, y_cross_clean)

    print(f"횡단면 모델 R²: {cross_score:.4f}")
    print("\\n표준화된 회귀계수:")
    for i, var in enumerate(X_vars):
        coef = cross_model.coef_[i]
        print(f"{var:15s}: {coef:7.3f}")

    # 3. 단순 회귀분석 (고용률 vs E9)
    print("\\n3. 단순 회귀분석 (고용률 vs E9)")

    # E9 체류자수와 고용률만
    X_simple = avg_df[['E9_체류자수']]
    y_simple = avg_df['고용률']

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

    # 회귀분석 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1) 산점도와 회귀선
    axes[0,0].scatter(X_simple_clean['E9_체류자수'], y_simple_clean, alpha=0.6, s=30)
    axes[0,0].plot(X_simple_clean['E9_체류자수'],
                   simple_model.predict(X_simple_clean),
                   'r-', linewidth=2, label=f'R² = {simple_score:.3f}')
    axes[0,0].set_xlabel('E9 체류자수 (5년 평균)')
    axes[0,0].set_ylabel('고용률 (5년 평균, %)')
    axes[0,0].set_title('고용률 vs E9 체류자수')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2) 제조업 비중 vs 고용률
    mask2 = ~(avg_df['제조업_비중'].isna() | avg_df['고용률'].isna())
    axes[0,1].scatter(avg_df[mask2]['제조업_비중'], avg_df[mask2]['고용률'], alpha=0.6, s=30)
    z = np.polyfit(avg_df[mask2]['제조업_비중'], avg_df[mask2]['고용률'], 1)
    p = np.poly1d(z)
    axes[0,1].plot(avg_df[mask2]['제조업_비중'], p(avg_df[mask2]['제조업_비중']), "r--", alpha=0.8)
    axes[0,1].set_xlabel('제조업 비중 (%)')
    axes[0,1].set_ylabel('고용률 (%)')
    axes[0,1].set_title('고용률 vs 제조업 비중')
    axes[0,1].grid(True, alpha=0.3)

    # 3) 종사자 밀도 vs 고용률
    mask3 = ~(avg_df['종사자_밀도'].isna() | avg_df['고용률'].isna() |
              np.isinf(avg_df['종사자_밀도']))
    axes[1,0].scatter(avg_df[mask3]['종사자_밀도'], avg_df[mask3]['고용률'], alpha=0.6, s=30)
    axes[1,0].set_xlabel('종사자 밀도 (명/km²)')
    axes[1,0].set_ylabel('고용률 (%)')
    axes[1,0].set_title('고용률 vs 종사자 밀도')
    axes[1,0].grid(True, alpha=0.3)

    # 4) 회귀계수 비교
    coefficients = [simple_coef * 1000, cross_model.coef_[0], cross_model.coef_[1], cross_model.coef_[2]]
    var_names = ['E9(단순)', 'E9(표준화)', '제조업비중(표준화)', '서비스업비중(표준화)']

    colors = ['red' if c > 0 else 'blue' for c in coefficients]
    bars = axes[1,1].bar(range(len(coefficients)), coefficients, color=colors, alpha=0.7)
    axes[1,1].set_xticks(range(len(var_names)))
    axes[1,1].set_xticklabels(var_names, rotation=45)
    axes[1,1].set_ylabel('회귀계수')
    axes[1,1].set_title('회귀계수 비교')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # 계수 값 표시
    for bar, coef in zip(bars, coefficients):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{coef:.3f}',
                       ha='center', va='bottom' if height > 0 else 'top')

    plt.tight_layout()
    plt.savefig('/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/regression_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 회귀분석 결과 저장
    regression_results = {
        'panel_r2': panel_score,
        'panel_e9_coef': panel_coef,
        'cross_r2': cross_score,
        'cross_coefficients': dict(zip(X_vars, cross_model.coef_)),
        'simple_r2': simple_score,
        'simple_coef': simple_coef,
        'simple_intercept': simple_intercept
    }

    return regression_results

def time_series_analysis(df):
    """시계열 분석"""
    print("\\n=== 시계열 분석 ===")

    # 연도별 평균 계산
    yearly_stats = df.groupby('연도').agg({
        '고용률': 'mean',
        'E9_체류자수': 'mean',
        '제조업_종사자수': 'sum',
        '전체_종사자수': 'sum'
    }).reset_index()

    # 제조업 비중 계산
    yearly_stats['제조업_비중'] = yearly_stats['제조업_종사자수'] / yearly_stats['전체_종사자수'] * 100

    # 시계열 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1) 고용률 추이
    axes[0,0].plot(yearly_stats['연도'], yearly_stats['고용률'], 'o-', linewidth=2, markersize=8)
    axes[0,0].set_title('연도별 평균 고용률 추이', fontweight='bold')
    axes[0,0].set_xlabel('연도')
    axes[0,0].set_ylabel('고용률 (%)')
    axes[0,0].grid(True, alpha=0.3)
    for i, row in yearly_stats.iterrows():
        axes[0,0].annotate(f'{row["고용률"]:.1f}%',
                          (row['연도'], row['고용률']),
                          textcoords="offset points", xytext=(0,10), ha='center')

    # 2) E9 체류자 수 추이
    axes[0,1].plot(yearly_stats['연도'], yearly_stats['E9_체류자수'], 'o-',
                   linewidth=2, markersize=8, color='red')
    axes[0,1].set_title('연도별 평균 E9 체류자 수 추이', fontweight='bold')
    axes[0,1].set_xlabel('연도')
    axes[0,1].set_ylabel('E9 체류자 수 (명)')
    axes[0,1].grid(True, alpha=0.3)
    for i, row in yearly_stats.iterrows():
        axes[0,1].annotate(f'{row["E9_체류자수"]:.0f}',
                          (row['연도'], row['E9_체류자수']),
                          textcoords="offset points", xytext=(0,10), ha='center')

    # 3) 제조업 비중 추이
    axes[1,0].plot(yearly_stats['연도'], yearly_stats['제조업_비중'], 'o-',
                   linewidth=2, markersize=8, color='green')
    axes[1,0].set_title('연도별 제조업 비중 추이', fontweight='bold')
    axes[1,0].set_xlabel('연도')
    axes[1,0].set_ylabel('제조업 비중 (%)')
    axes[1,0].grid(True, alpha=0.3)
    for i, row in yearly_stats.iterrows():
        axes[1,0].annotate(f'{row["제조업_비중"]:.1f}%',
                          (row['연도'], row['제조업_비중']),
                          textcoords="offset points", xytext=(0,10), ha='center')

    # 4) 고용률 vs E9 상관관계 (연도별)
    corr_by_year = []
    for year in sorted(df['연도'].unique()):
        year_data = df[df['연도'] == year]
        if len(year_data) > 10:  # 충분한 데이터가 있는 경우만
            corr, p_val = stats.pearsonr(year_data['고용률'].dropna(),
                                        year_data['E9_체류자수'].dropna())
            corr_by_year.append({'연도': year, '상관계수': corr, 'p_value': p_val})

    if corr_by_year:
        corr_df = pd.DataFrame(corr_by_year)
        bars = axes[1,1].bar(corr_df['연도'], corr_df['상관계수'],
                            color=['red' if x > 0 else 'blue' for x in corr_df['상관계수']],
                            alpha=0.7)
        axes[1,1].set_title('연도별 고용률-E9 상관계수', fontweight='bold')
        axes[1,1].set_xlabel('연도')
        axes[1,1].set_ylabel('상관계수')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # 상관계수 값 표시
        for bar, corr_val in zip(bars, corr_df['상관계수']):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{corr_val:.3f}',
                          ha='center', va='bottom' if height > 0 else 'top')

    plt.tight_layout()
    plt.savefig('/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/time_series_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return yearly_stats

def descriptive_statistics(avg_df):
    """기술통계 분석"""
    print("\\n=== 기술통계 분석 ===")

    # 주요 변수들의 기술통계
    key_vars = ['고용률', 'E9_체류자수', '제조업_비중', '서비스업_비중', '종사자_밀도']
    desc_stats = avg_df[key_vars].describe()

    print("기술통계 (5년 평균 기준):")
    print(desc_stats.round(2))

    # 상위/하위 지역 분석
    print("\\n=== 상위/하위 지역 분석 ===")

    # 고용률 상위 10개 지역
    top_employment = avg_df.nlargest(10, '고용률')[['시도', '시군구', '고용률', 'E9_체류자수']]
    print("\\n고용률 상위 10개 지역:")
    print(top_employment.round(2))

    # E9 체류자 수 상위 10개 지역
    top_e9 = avg_df.nlargest(10, 'E9_체류자수')[['시도', '시군구', '고용률', 'E9_체류자수']]
    print("\\nE9 체류자 수 상위 10개 지역:")
    print(top_e9.round(2))

    # 분포 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1) 고용률 분포
    axes[0,0].hist(avg_df['고용률'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(avg_df['고용률'].mean(), color='red', linestyle='--',
                      label=f'평균: {avg_df["고용률"].mean():.1f}%')
    axes[0,0].set_title('고용률 분포', fontweight='bold')
    axes[0,0].set_xlabel('고용률 (%)')
    axes[0,0].set_ylabel('지자체 수')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2) E9 체류자 수 분포 (로그 스케일)
    e9_positive = avg_df[avg_df['E9_체류자수'] > 0]['E9_체류자수']
    axes[0,1].hist(np.log1p(e9_positive), bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].axvline(np.log1p(e9_positive).mean(), color='red', linestyle='--',
                      label=f'평균(log): {np.log1p(e9_positive).mean():.1f}')
    axes[0,1].set_title('E9 체류자 수 분포 (log scale)', fontweight='bold')
    axes[0,1].set_xlabel('log(E9 체류자수 + 1)')
    axes[0,1].set_ylabel('지자체 수')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3) 제조업 비중 분포
    manufacturing_clean = avg_df[avg_df['제조업_비중'].notna()]['제조업_비중']
    axes[1,0].hist(manufacturing_clean, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1,0].axvline(manufacturing_clean.mean(), color='red', linestyle='--',
                      label=f'평균: {manufacturing_clean.mean():.1f}%')
    axes[1,0].set_title('제조업 비중 분포', fontweight='bold')
    axes[1,0].set_xlabel('제조업 비중 (%)')
    axes[1,0].set_ylabel('지자체 수')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4) 박스플롯 - 시도별 고용률
    major_sido = avg_df['시도'].value_counts().head(8).index
    sido_employment = []
    sido_labels = []
    for sido in major_sido:
        sido_data = avg_df[avg_df['시도'] == sido]['고용률'].dropna()
        if len(sido_data) > 0:
            sido_employment.append(sido_data)
            sido_labels.append(sido.replace('특별시', '').replace('광역시', '').replace('특별자치도', '').replace('도', ''))

    if sido_employment:
        axes[1,1].boxplot(sido_employment, labels=sido_labels)
        axes[1,1].set_title('시도별 고용률 분포', fontweight='bold')
        axes[1,1].set_xlabel('시도')
        axes[1,1].set_ylabel('고용률 (%)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/descriptive_statistics.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return desc_stats, top_employment, top_e9

def main():
    """메인 실행 함수"""
    print("통계 분석 시작...\\n")

    # 데이터 로드
    df = load_data()

    # 5년 평균 계산
    avg_df = calculate_averages(df)

    # 분석 실행
    correlation_matrix = correlation_analysis(avg_df)
    regression_results = regression_analysis(df, avg_df)
    yearly_stats = time_series_analysis(df)
    desc_stats, top_employment, top_e9 = descriptive_statistics(avg_df)

    # 결과 저장
    results = {
        'correlation_matrix': correlation_matrix,
        'regression_results': regression_results,
        'yearly_stats': yearly_stats,
        'descriptive_stats': desc_stats,
        'top_employment_regions': top_employment,
        'top_e9_regions': top_e9,
        'average_data': avg_df
    }

    # CSV 파일들 저장
    avg_df.to_csv('/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/average_statistics.csv',
                  index=False, encoding='utf-8')

    yearly_stats.to_csv('/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/yearly_trends.csv',
                        index=False, encoding='utf-8')

    correlation_matrix.to_csv('/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/correlation_matrix.csv',
                              encoding='utf-8')

    print("\\n=== 분석 완료 ===")
    print("결과 파일들이 result_data 폴더에 저장되었습니다:")
    print("- correlation_heatmap.png: 상관관계 히트맵")
    print("- regression_analysis.png: 회귀분석 결과")
    print("- time_series_analysis.png: 시계열 분석")
    print("- descriptive_statistics.png: 기술통계 분석")
    print("- average_statistics.csv: 5년 평균 통계")
    print("- yearly_trends.csv: 연도별 추이")
    print("- correlation_matrix.csv: 상관계수 매트릭스")

    return results

if __name__ == "__main__":
    results = main()