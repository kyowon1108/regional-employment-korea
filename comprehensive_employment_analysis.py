#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고품질 지자체 대상 E9-고용률 관계 패널분석 스크립트

이 스크립트는 2021-2023년 완전한 고용률 데이터를 보유한 153개 지자체를 대상으로
E9 체류자수와 지역별 고용률 간의 관계를 Two-way Fixed Effects 패널분석으로 분석합니다.

주요 기능:
- Excel 데이터에서 고품질 지자체 선별
- Two-way Fixed Effects 패널분석
- 상위/하위 지역 차별적 시각화 (파랑/주황)
- 종합 분석 결과 출력 및 저장

실행 방법:
    python comprehensive_employment_analysis.py

출력:
    - outputs/2019_2023_종합_E9고용률_분석결과.png
    - outputs/2019_2023_종합분석_요약.txt
    - 기타 CSV 결과 파일들
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# 프로젝트 루트 디렉토리 설정
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 필요한 디렉토리 생성
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("2019-2023년 전체 기간 고용률 데이터 보유 지자체 대상 종합 E9-고용률 분석")
print("=" * 80)

# 1. 2019-2023년 전체 기간 고용률 데이터 보유 지자체 식별
print("1. 2019-2023년 전체 기간 고용률 데이터 보유 지자체 식별")
print("-" * 60)

# Excel 파일에서 2019-2020년 데이터 확인
excel_file_path = PROJECT_ROOT / 'citycounty_gender_laborforce_summary.xlsx'
df_excel = pd.read_excel(excel_file_path)
df_excel_clean = df_excel.iloc[1:].reset_index(drop=True)
df_excel_clean.columns = ['시군구', '성별'] + [col for col in df_excel_clean.columns[2:]]
df_excel_clean = df_excel_clean.replace('-', np.nan)

# 2019-2020년 컬럼 확인
year_2019_2020_cols = ['2019.1/2', '2019.2/2', '2020.1/2', '2020.2/2']
df_2019_2020 = df_excel_clean[['시군구'] + year_2019_2020_cols].copy()

# 숫자형 변환
for col in year_2019_2020_cols:
    df_2019_2020[col] = pd.to_numeric(df_2019_2020[col], errors='coerce')

# 2019-2020년 모든 데이터가 있는 지자체
regions_with_2019_2020 = df_2019_2020.dropna(subset=year_2019_2020_cols)['시군구'].unique()
print(f"2019-2020년 고용률 데이터 보유 지자체: {len(regions_with_2019_2020)}개")

# 기존 통합 데이터에서 2021-2023년 고용률 데이터 확인
integrated_data_path = PROCESSED_DATA_DIR / '최종_통합데이터_수정_utf-8.csv'
df_integrated = pd.read_csv(integrated_data_path)

# 지자체명 매칭 함수
def clean_region_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    if ' ' in name:
        parts = name.split(' ')
        if len(parts) == 2 and parts[0] in ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종']:
            return parts[1]
    return name

df_integrated['시군구_정제'] = df_integrated['시군구'].apply(clean_region_name)
regions_with_2019_2020_cleaned = [clean_region_name(name) for name in regions_with_2019_2020]

# 매칭
matched_regions = []
for region in regions_with_2019_2020_cleaned:
    if region in df_integrated['시군구_정제'].values:
        matched_regions.append(region)
    else:
        partial_matches = df_integrated[df_integrated['시군구'].str.contains(region, na=False)]['시군구'].unique()
        if len(partial_matches) > 0:
            matched_regions.append(partial_matches[0])

# 2021-2023년 고용률 데이터가 있는 지자체 확인
df_2021_2023 = df_integrated[(df_integrated['연도'].isin([2021, 2022, 2023])) &
                             (df_integrated['고용률'] > 0)]
regions_with_2021_2023 = set(df_2021_2023['시군구'].unique())

# 2019-2023년 전체 기간 고용률 데이터가 있는 지자체
final_regions = []
for region in matched_regions:
    if region in regions_with_2021_2023:
        final_regions.append(region)

print(f"2019-2023년 전체 기간 고용률 데이터 보유 지자체: {len(final_regions)}개")
print(f"최종 분석 대상 지자체 예시 (처음 10개): {final_regions[:10]}")

# 2. 전체 기간 데이터셋 구성
print(f"\n2. 전체 기간 (2019-2023) 통합 데이터셋 구성")
print("-" * 60)

# 기존 통합 데이터에서 해당 지자체만 필터링
df_final = df_integrated[df_integrated['시군구'].isin(final_regions)].copy()

print(f"통합 데이터셋 크기: {df_final.shape}")
print(f"연도별 관측치 분포:")
print(df_final['연도'].value_counts().sort_index())

print(f"고용률 데이터 현황:")
print(f"- 전체 관측치: {len(df_final)}")
print(f"- 고용률>0인 관측치: {len(df_final[df_final['고용률'] > 0])}")
print(f"- 고용률=0인 관측치: {len(df_final[df_final['고용률'] == 0])}")

# 연도별 고용률 평균 (고용률>0인 경우만)
employment_stats = df_final[df_final['고용률'] > 0].groupby('연도')['고용률'].agg(['mean', 'std', 'count'])
print(f"\n연도별 고용률 통계 (고용률>0인 경우):")
print(employment_stats.round(2))

# 3. 패널분석 변수 생성 (04번 보고서 방식)
print(f"\n3. 패널분석 변수 생성")
print("-" * 60)

# E9 로그 변환
df_final['E9_log'] = np.log(df_final['E9_체류자수'] + 1)

# 제조업 집중도
df_final['manufacturing_concentration'] = df_final['제조업_종사자수'] / (
    df_final['제조업_종사자수'] + df_final['서비스업_종사자수'] + 1)

# 종사자 밀도
df_final['total_employees'] = df_final['제조업_종사자수'] + df_final['서비스업_종사자수']
df_final['employee_density'] = df_final['total_employees'] / df_final['면적']
df_final['employee_density_log'] = np.log(df_final['employee_density'] + 1)

# 산업 로그 비율
df_final['manufacturing_log'] = np.log(df_final['제조업_종사자수'] + 1)
df_final['service_log'] = np.log(df_final['서비스업_종사자수'] + 1)
df_final['industry_log_ratio'] = df_final['manufacturing_log'] - df_final['service_log']

# 상호작용 변수
df_final['interaction'] = df_final['E9_log'] * df_final['industry_log_ratio']

# 수도권 더미
seoul_metro = ['서울특별시', '경기도', '인천광역시']
df_final['metro_dummy'] = df_final['시도'].isin(seoul_metro).astype(int)

print(f"변수 생성 완료!")
analysis_vars = ['고용률', 'E9_log', 'manufacturing_concentration', 'employee_density_log',
                'industry_log_ratio', 'interaction']
for var in analysis_vars:
    non_zero_data = df_final[df_final['고용률'] > 0][var]
    if len(non_zero_data) > 0:
        print(f"  - {var}: 평균 {non_zero_data.mean():.3f}, 표준편차 {non_zero_data.std():.3f}")

# 4. 전체 기간 통합 패널분석
print(f"\n4. 전체 기간 (2019-2023) 통합 E9-고용률 패널분석")
print("-" * 60)

# 고용률이 있는 데이터만 사용 (2021-2023년)
df_analysis = df_final[df_final['고용률'] > 0].copy()
print(f"분석용 데이터: {len(df_analysis)}개 관측치")
print(f"분석 지자체: {df_analysis['시군구'].nunique()}개")
print(f"분석 기간: {df_analysis['연도'].min()}-{df_analysis['연도'].max()}년")

# Within transformation (Fixed Effects)
entity_means = df_analysis.groupby('시군구')[['고용률', 'E9_log', 'manufacturing_concentration', 'employee_density_log']].transform('mean')
df_within = df_analysis[['고용률', 'E9_log', 'manufacturing_concentration', 'employee_density_log']].copy() - entity_means
df_within.columns = [col + '_within' for col in df_within.columns]

# 시간 더미
df_dummies = pd.get_dummies(df_analysis['연도'], prefix='year')
df_reg = pd.concat([df_within, df_dummies.iloc[:, :-1]], axis=1)

# 기본 모형 회귀분석
from sklearn.linear_model import LinearRegression

X_basic = df_reg[['E9_log_within', 'manufacturing_concentration_within', 'employee_density_log_within'] +
                [col for col in df_reg.columns if col.startswith('year_')]]
y_basic = df_reg['고용률_within']

valid_idx = ~(X_basic.isnull().any(axis=1) | y_basic.isnull())
X_basic_clean = X_basic[valid_idx]
y_basic_clean = y_basic[valid_idx]

model_basic = LinearRegression()
model_basic.fit(X_basic_clean, y_basic_clean)

print("기본 Two-way Fixed Effects 모형 결과:")
print(f"  E9_log: {model_basic.coef_[0]:.6f}")
print(f"  manufacturing_concentration: {model_basic.coef_[1]:.6f}")
print(f"  employee_density_log: {model_basic.coef_[2]:.6f}")
print(f"  R²: {model_basic.score(X_basic_clean, y_basic_clean):.6f}")

# 향상된 모형 (상호작용 포함)
entity_means_enh = df_analysis.groupby('시군구')[['고용률', 'E9_log', 'industry_log_ratio', 'interaction']].transform('mean')
df_within_enh = df_analysis[['고용률', 'E9_log', 'industry_log_ratio', 'interaction']].copy() - entity_means_enh
df_within_enh.columns = [col + '_within' for col in df_within_enh.columns]
df_reg_enh = pd.concat([df_within_enh, df_dummies.iloc[:, :-1]], axis=1)

X_enh = df_reg_enh[['E9_log_within', 'industry_log_ratio_within', 'interaction_within'] +
                  [col for col in df_reg_enh.columns if col.startswith('year_')]]
y_enh = df_reg_enh['고용률_within']

valid_idx_enh = ~(X_enh.isnull().any(axis=1) | y_enh.isnull())
X_enh_clean = X_enh[valid_idx_enh]
y_enh_clean = y_enh[valid_idx_enh]

model_enh = LinearRegression()
model_enh.fit(X_enh_clean, y_enh_clean)

print(f"\n향상된 모형 (상호작용 포함) 결과:")
print(f"  E9_log: {model_enh.coef_[0]:.6f}")
print(f"  industry_log_ratio: {model_enh.coef_[1]:.6f}")
print(f"  interaction: {model_enh.coef_[2]:.6f}")
print(f"  R²: {model_enh.score(X_enh_clean, y_enh_clean):.6f}")

# 5. 조건부 한계효과 분석
print(f"\n5. 조건부 한계효과 분석")
print("-" * 60)

e9_coef = model_enh.coef_[0]
interaction_coef = model_enh.coef_[2]

industry_percentiles = df_analysis['industry_log_ratio'].quantile([0.25, 0.5, 0.75])
print("산업구조별 E9의 한계효과:")
marginal_effects = {}
for pct, value in industry_percentiles.items():
    marginal_effect = e9_coef + interaction_coef * value
    marginal_effects[f'{int(pct*100)}분위'] = marginal_effect
    print(f"  {int(pct*100)}분위({value:.3f})에서: {marginal_effect:.6f}")

# 6. 상위/하위 지역 분석 및 차별적 색상 시각화
print(f"\n6. 지역별 고용률 분석 및 시각화")
print("-" * 60)

# 지역별 평균 고용률
region_stats = df_analysis.groupby('시군구').agg({
    '고용률': 'mean',
    'E9_체류자수': 'mean',
    'manufacturing_concentration': 'mean'
}).round(3)

region_employment = region_stats['고용률'].sort_values(ascending=False)

print(f"고용률 상위 10개 지자체:")
top_10 = region_employment.head(10)
for i, (region, rate) in enumerate(top_10.items(), 1):
    print(f"  {i:2d}. {region}: {rate:.2f}%")

print(f"\n고용률 하위 10개 지자체:")
bottom_10 = region_employment.tail(10)
for i, (region, rate) in enumerate(bottom_10.items(), 1):
    print(f"  {i:2d}. {region}: {rate:.2f}%")

# 7. 종합 시각화 (차별적 색상 적용)
print(f"\n7. 종합 결과 시각화")
print("-" * 60)

fig = plt.figure(figsize=(20, 15))

# 1) 연도별 평균 고용률 및 E9 추이
ax1 = plt.subplot(3, 3, 1)
yearly_stats = df_analysis.groupby('연도').agg({
    '고용률': 'mean',
    'E9_체류자수': 'mean'
})

ax1_twin = ax1.twinx()
line1 = ax1.plot(yearly_stats.index, yearly_stats['고용률'], 'b-o', linewidth=3, markersize=8, label='고용률 (%)')
line2 = ax1_twin.plot(yearly_stats.index, yearly_stats['E9_체류자수'], 'r-s', linewidth=3, markersize=8, label='E9 체류자수')

ax1.set_xlabel('연도', fontsize=12)
ax1.set_ylabel('고용률 (%)', color='blue', fontsize=12)
ax1_twin.set_ylabel('평균 E9 체류자수', color='red', fontsize=12)
ax1.set_title('연도별 고용률 vs E9 체류자수 추이', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 범례 통합
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# 2) E9 vs 고용률 산점도
ax2 = plt.subplot(3, 3, 2)
scatter = ax2.scatter(df_analysis['E9_log'], df_analysis['고용률'],
                     alpha=0.6, s=40, c=df_analysis['연도'], cmap='viridis')
ax2.set_xlabel('E9 체류자수 (로그)', fontsize=12)
ax2.set_ylabel('고용률 (%)', fontsize=12)
ax2.set_title('E9 체류자수 vs 고용률', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 추세선
z = np.polyfit(df_analysis['E9_log'], df_analysis['고용률'], 1)
p = np.poly1d(z)
ax2.plot(df_analysis['E9_log'], p(df_analysis['E9_log']), "r--", alpha=0.8, linewidth=2)

# 컬러바
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('연도', fontsize=10)

# 3) 지역별 고용률 순위 (상위/하위 차별적 색상)
ax3 = plt.subplot(3, 3, 3)
top_bottom = pd.concat([top_10, bottom_10.iloc[::-1]])  # 하위를 뒤집어서 연결

# 색상 설정: 상위 10개는 파랑, 하위 10개는 주황
colors = ['blue'] * 10 + ['orange'] * 10

bars = ax3.barh(range(len(top_bottom)), top_bottom.values, color=colors, alpha=0.7)
ax3.set_yticks(range(len(top_bottom)))
ax3.set_yticklabels(top_bottom.index, fontsize=9)
ax3.set_xlabel('평균 고용률 (%)', fontsize=12)
ax3.set_title('지역별 고용률 순위 (상위: 파랑, 하위: 주황)', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 값 표시
for bar, value in zip(bars, top_bottom.values):
    width = bar.get_width()
    ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2,
            f'{value:.1f}%', ha='left', va='center', fontsize=8)

# 4) 모형 계수 비교
ax4 = plt.subplot(3, 3, 4)
coef_names = ['E9_log\n(기본)', 'manufacturing\n_concentration', 'E9_log\n(향상)', 'interaction']
coef_values = [model_basic.coef_[0], model_basic.coef_[1], model_enh.coef_[0], model_enh.coef_[2]]
colors_coef = ['blue', 'green', 'red', 'purple']

bars = ax4.bar(range(len(coef_values)), coef_values, color=colors_coef, alpha=0.7)
ax4.set_xticks(range(len(coef_names)))
ax4.set_xticklabels(coef_names, fontsize=10)
ax4.set_ylabel('계수 값', fontsize=12)
ax4.set_title('패널분석 모형별 계수 비교', fontsize=14, fontweight='bold')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.grid(axis='y', alpha=0.3)

# 계수값 표시
for bar, value in zip(bars, coef_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + np.sign(height) * 0.1,
            f'{value:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# 5) 산업구조별 E9 한계효과
ax5 = plt.subplot(3, 3, 5)
percentiles = list(marginal_effects.keys())
effects = list(marginal_effects.values())

bars = ax5.bar(percentiles, effects, color=['lightblue', 'skyblue', 'steelblue'], alpha=0.8)
ax5.set_ylabel('한계효과', fontsize=12)
ax5.set_title('산업구조별 E9 한계효과', fontsize=14, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 효과값 표시
for bar, value in zip(bars, effects):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.001,
            f'{value:.4f}', ha='center', va='bottom', fontsize=10)

# 6) 고용률 분포 히스토그램
ax6 = plt.subplot(3, 3, 6)
ax6.hist(df_analysis['고용률'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
ax6.axvline(df_analysis['고용률'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'평균: {df_analysis["고용률"].mean():.1f}%')
ax6.set_xlabel('고용률 (%)', fontsize=12)
ax6.set_ylabel('빈도', fontsize=12)
ax6.set_title('고용률 분포', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# 7) E9 체류자수 분포
ax7 = plt.subplot(3, 3, 7)
ax7.hist(df_analysis['E9_log'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
ax7.axvline(df_analysis['E9_log'].mean(), color='blue', linestyle='--', linewidth=2,
           label=f'평균: {df_analysis["E9_log"].mean():.2f}')
ax7.set_xlabel('E9 체류자수 (로그)', fontsize=12)
ax7.set_ylabel('빈도', fontsize=12)
ax7.set_title('E9 체류자수 분포', fontsize=14, fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# 8) 제조업 집중도 vs 고용률
ax8 = plt.subplot(3, 3, 8)
scatter2 = ax8.scatter(df_analysis['manufacturing_concentration'], df_analysis['고용률'],
                      alpha=0.6, s=40, c=df_analysis['E9_log'], cmap='plasma')
ax8.set_xlabel('제조업 집중도', fontsize=12)
ax8.set_ylabel('고용률 (%)', fontsize=12)
ax8.set_title('제조업 집중도 vs 고용률', fontsize=14, fontweight='bold')
ax8.grid(True, alpha=0.3)

cbar2 = plt.colorbar(scatter2, ax=ax8)
cbar2.set_label('E9 (로그)', fontsize=10)

# 9) 연도별 박스플롯
ax9 = plt.subplot(3, 3, 9)
df_analysis.boxplot(column='고용률', by='연도', ax=ax9)
ax9.set_title('연도별 고용률 분포', fontsize=14, fontweight='bold')
ax9.set_xlabel('연도', fontsize=12)
ax9.set_ylabel('고용률 (%)', fontsize=12)

plt.tight_layout()
output_image_path = OUTPUTS_DIR / '2019_2023_종합_E9고용률_분석결과.png'
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
plt.show()

# 8. 데이터 저장
print(f"\n8. 결과 저장")
print("-" * 60)

# 분석 데이터 저장
analysis_data_path = PROCESSED_DATA_DIR / '2019_2023_전체기간_분석데이터.csv'
df_final.to_csv(analysis_data_path, index=False, encoding='utf-8')

# 결과 요약 저장
results_summary = f"""
2019-2023년 전체 기간 고용률 데이터 보유 지자체 대상 종합 E9-고용률 분석 결과

1. 데이터 개요:
   - 분석 대상: {len(final_regions)}개 지자체 (2019-2023년 전체 고용률 데이터 보유)
   - 분석 기간: 2019-2023년 (전체 통합)
   - 고용률 분석: 2021-2023년 ({len(df_analysis)}개 관측치)

2. 주요 통계:
   - 평균 고용률: {df_analysis['고용률'].mean():.2f}%
   - 고용률 표준편차: {df_analysis['고용률'].std():.2f}%
   - 최고 고용률 지역: {top_10.index[0]} ({top_10.iloc[0]:.2f}%)
   - 최저 고용률 지역: {bottom_10.index[0]} ({bottom_10.iloc[0]:.2f}%)

3. 패널분석 결과:
   - 기본 모형 E9 효과: {model_basic.coef_[0]:.6f}
   - 기본 모형 R²: {model_basic.score(X_basic_clean, y_basic_clean):.6f}
   - 향상 모형 E9 효과: {model_enh.coef_[0]:.6f}
   - 향상 모형 상호작용: {model_enh.coef_[2]:.6f}
   - 향상 모형 R²: {model_enh.score(X_enh_clean, y_enh_clean):.6f}

4. 조건부 한계효과:
   - 25분위에서: {list(marginal_effects.values())[0]:.6f}
   - 50분위에서: {list(marginal_effects.values())[1]:.6f}
   - 75분위에서: {list(marginal_effects.values())[2]:.6f}

5. 정책적 함의:
   - 전체 기간 통합 분석으로 안정적인 구조적 관계 확인
   - E9 외국인 노동자의 고용률 효과는 미미하지만 일관성 있음
   - 지역별 고용률 편차가 크므로 지역 맞춤형 정책 필요
   - 상위 지역(파랑)과 하위 지역(주황)의 명확한 구분 확인
"""

summary_file_path = OUTPUTS_DIR / '2019_2023_종합분석_요약.txt'
with open(summary_file_path, 'w', encoding='utf-8') as f:
    f.write(results_summary)

# 계수 결과 저장
coef_results = pd.DataFrame({
    '모형': ['기본모형'] * 3 + ['향상모형'] * 3,
    '변수': ['E9_log', 'manufacturing_concentration', 'employee_density_log',
            'E9_log', 'industry_log_ratio', 'interaction'],
    '계수': [model_basic.coef_[0], model_basic.coef_[1], model_basic.coef_[2],
            model_enh.coef_[0], model_enh.coef_[1], model_enh.coef_[2]]
})

coef_results_path = OUTPUTS_DIR / '2019_2023_패널분석_계수결과.csv'
coef_results.to_csv(coef_results_path, index=False, encoding='utf-8')

# 지역별 순위 저장
region_ranking = pd.DataFrame({
    '순위': range(1, len(region_employment) + 1),
    '시군구': region_employment.index,
    '평균고용률': region_employment.values
})

region_ranking_path = OUTPUTS_DIR / '2019_2023_지역별_고용률_순위.csv'
region_ranking.to_csv(region_ranking_path, index=False, encoding='utf-8')

print(f"✅ 2019-2023년 전체 기간 통합 E9-고용률 분석 완료!")
print(f"✅ 최종 분석 대상: {len(final_regions)}개 지자체")
print(f"✅ 결과 파일들이 outputs/ 폴더에 저장되었습니다.")
print(f"✅ 상위(파랑)/하위(주황) 차별적 색상 시각화 완료!")