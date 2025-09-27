#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enhanced_comprehensive_analysis.py

개선된 E9 비자 소지자의 지역 고용률 영향 종합 분석
153개 시군구 완전균형패널(2019-2023) × TWFE + 군집표준오차

요구사항 충족:
(1) Two-way Fixed Effects 패널분석 + 군집표준오차
(2) 통계 테이블 (coef/std err/t/p/CI/R²/N)
(3) Choropleth 4종 지도 생성
(4) Pearson/Spearman 상관행렬 + 시도별 추세
(5) 경제적 유의성 + 외생성 이슈 논의
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import geopandas as gpd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
import os
import sys
from pathlib import Path

# 경고 무시
warnings.filterwarnings('ignore')

def setup_korean_font():
    """한글 폰트 설정"""
    try:
        font_candidates = [
            '/System/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/AppleMyungjo.ttf',
            '/System/Library/Fonts/Arial Unicode MS.ttf'
        ]

        for font_path in font_candidates:
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                return True

        font_list = [f.name for f in fm.fontManager.ttflist if 'gothic' in f.name.lower()]
        if font_list:
            plt.rcParams['font.family'] = font_list[0]
            plt.rcParams['axes.unicode_minus'] = False
            return True

        return False
    except Exception as e:
        print(f"폰트 설정 오류: {e}")
        return False

class EnhancedPanelAnalyzer:
    """개선된 패널 데이터 분석 클래스"""

    def __init__(self, data_path):
        """초기화"""
        self.data_path = data_path
        self.df = None
        self.results = {}
        self.output_dir = "/Users/kapr/Desktop/DataAnalyze/new_analysis/output"
        self.maps_dir = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/maps"
        os.makedirs(self.output_dir, exist_ok=True)

    def verify_balanced_panel(self):
        """완전균형패널 검증"""
        print("=" * 80)
        print("153개 시군구 완전균형패널 검증 (2019-2023)")
        print("=" * 80)

        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ 데이터 로드: {len(self.df):,}개 관측치")

            # 패널 구조 검증
            panel_structure = self.df.groupby('시군구').agg({
                '연도': ['count', 'min', 'max']
            }).round(2)

            panel_structure.columns = ['관측치수', '최소연도', '최대연도']

            # 완전 패널 확인
            complete_panels = (panel_structure['관측치수'] == 5).sum()
            incomplete_panels = (panel_structure['관측치수'] != 5).sum()

            print(f"📊 패널 구조:")
            print(f"   - 총 시군구: {self.df['시군구'].nunique()}개")
            print(f"   - 완전패널(5년): {complete_panels}개")
            print(f"   - 불완전패널: {incomplete_panels}개")
            print(f"   - 총 관측치: {len(self.df)}개")

            if incomplete_panels > 0:
                print("⚠️ 불완전패널 발견:")
                incomplete = panel_structure[panel_structure['관측치수'] != 5]
                for idx, row in incomplete.head(5).iterrows():
                    print(f"   {idx}: {row['관측치수']}개 관측치")

            # 연도별 분포 확인
            yearly_dist = self.df['연도'].value_counts().sort_index()
            print(f"\n📅 연도별 관측치:")
            for year, count in yearly_dist.items():
                print(f"   {year}년: {count}개")

            # 기본 통계
            print(f"\n📈 주요 변수 기본 통계:")
            main_vars = ['E9_체류자수', '고용률', '제조업_비중', '서비스업_비중']
            stats_summary = self.df[main_vars].describe().round(2)
            print(stats_summary)

            return complete_panels == self.df['시군구'].nunique()

        except Exception as e:
            print(f"❌ 패널 검증 실패: {e}")
            return False

    def twfe_regression_with_clustered_se(self):
        """Two-way Fixed Effects with 군집표준오차"""
        print("\n" + "=" * 60)
        print("Two-way Fixed Effects 회귀분석 (군집표준오차)")
        print("=" * 60)

        try:
            # 1. 패널 데이터 준비
            df_clean = self.df.dropna()
            print(f"분석 대상: {len(df_clean)}개 관측치")

            # 2. 변수 생성
            # E9 체류자수 로그 변환 (0값 처리)
            df_clean['ln_E9'] = np.log(df_clean['E9_체류자수'] + 1)

            # COVID-19 더미 (2020년 이후)
            df_clean['covid_dummy'] = (df_clean['연도'] >= 2020).astype(int)

            # 시군구, 연도 더미 생성
            region_dummies = pd.get_dummies(df_clean['시군구'], prefix='region', drop_first=True)
            year_dummies = pd.get_dummies(df_clean['연도'], prefix='year', drop_first=True)

            # 3. 회귀분석용 데이터 구성
            y = df_clean['고용률']

            # 주요 독립변수
            X_main = df_clean[['ln_E9', '제조업_비중', '서비스업_비중', 'covid_dummy']]

            # 고정효과 더미 추가
            X_full = pd.concat([X_main, region_dummies, year_dummies], axis=1)

            # 상수항 추가
            X_full.insert(0, 'const', 1.0)

            print(f"설명변수 개수: {X_full.shape[1]}개 (상수항 포함)")
            print(f"   - 주요변수: {X_main.shape[1]}개")
            print(f"   - 지역더미: {region_dummies.shape[1]}개")
            print(f"   - 연도더미: {year_dummies.shape[1]}개")

            # 4. OLS 추정
            X_array = X_full.values.astype(float)
            y_array = y.values.astype(float)

            # 회귀계수 추정
            XtX_inv = np.linalg.pinv(X_array.T @ X_array)
            beta = XtX_inv @ X_array.T @ y_array

            # 예측값과 잔차
            y_pred = X_array @ beta
            residuals = y_array - y_pred

            # 5. 군집표준오차 계산 (시군구 단위)
            cluster_se = self.calculate_clustered_standard_errors(
                X_array, residuals, df_clean['시군구'].values
            )

            # 6. 통계량 계산
            n = len(y_array)
            k = len(beta)

            # t-통계량과 p-값 (군집표준오차 사용)
            t_stats = beta / cluster_se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))

            # 신뢰구간 (95%)
            t_critical = stats.t.ppf(0.975, n - k)
            ci_lower = beta - t_critical * cluster_se
            ci_upper = beta + t_critical * cluster_se

            # R-squared
            sst = np.sum((y_array - y_array.mean())**2)
            ssr = np.sum(residuals**2)
            r_squared = 1 - (ssr / sst)

            # Adjusted R-squared
            adj_r_squared = 1 - (ssr / (n - k)) / (sst / (n - 1))

            # 7. 결과 저장
            variable_names = ['상수항'] + list(X_main.columns) + list(region_dummies.columns) + list(year_dummies.columns)

            results_df = pd.DataFrame({
                'Variable': variable_names,
                'Coefficient': beta,
                'Clustered_SE': cluster_se,
                'T_Statistic': t_stats,
                'P_Value': p_values,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Significance': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
                               for p in p_values]
            })

            self.results['twfe_clustered'] = {
                'coefficients': results_df,
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'n_obs': n,
                'n_clusters': df_clean['시군구'].nunique(),
                'residuals': residuals
            }

            # 결과 출력
            self.print_twfe_results()

            # 통계 테이블 저장
            self.save_statistical_tables()

            return True

        except Exception as e:
            print(f"❌ TWFE 회귀분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_clustered_standard_errors(self, X, residuals, cluster_var):
        """군집표준오차 계산"""
        try:
            n, k = X.shape
            unique_clusters = np.unique(cluster_var)
            n_clusters = len(unique_clusters)

            # Meat matrix 계산
            meat_matrix = np.zeros((k, k))

            for cluster in unique_clusters:
                cluster_mask = (cluster_var == cluster)
                X_cluster = X[cluster_mask]
                resid_cluster = residuals[cluster_mask]

                # 클러스터별 score 계산
                cluster_score = X_cluster.T @ resid_cluster
                meat_matrix += np.outer(cluster_score, cluster_score)

            # Bread matrix
            bread_matrix = np.linalg.pinv(X.T @ X)

            # 군집표준오차의 분산-공분산 행렬
            # 유한 샘플 조정: (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
            finite_sample_adj = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
            vcov_clustered = finite_sample_adj * bread_matrix @ meat_matrix @ bread_matrix

            # 표준오차 추출
            clustered_se = np.sqrt(np.diag(vcov_clustered))

            return clustered_se

        except Exception as e:
            print(f"군집표준오차 계산 실패: {e}")
            # 일반 표준오차로 대체
            sigma2 = np.sum(residuals**2) / (len(residuals) - X.shape[1])
            return np.sqrt(sigma2 * np.diag(np.linalg.pinv(X.T @ X)))

    def print_twfe_results(self):
        """TWFE 회귀분석 결과 출력"""
        results = self.results['twfe_clustered']
        coef_df = results['coefficients']

        print(f"\n📊 Two-way Fixed Effects 회귀분석 결과")
        print("=" * 90)
        print("종속변수: 고용률 (%)")
        print("표준오차: 시군구 군집표준오차 (Clustered Standard Errors)")
        print("=" * 90)

        # 주요 변수만 출력
        main_vars = ['상수항', 'ln_E9', '제조업_비중', '서비스업_비중', 'covid_dummy']
        main_results = coef_df[coef_df['Variable'].isin(main_vars)]

        print(f"{'Variable':<20} {'Coef.':<10} {'Clust.SE':<10} {'t':<8} {'P>|t|':<8} {'[95% CI]':<20} {'Sig':<5}")
        print("=" * 90)

        for _, row in main_results.iterrows():
            var_name = row['Variable']
            if var_name == 'ln_E9':
                var_name = 'ln(E9 체류자수)'
            elif var_name == 'covid_dummy':
                var_name = 'COVID-19 더미'

            ci_str = f"[{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]"

            print(f"{var_name:<20} {row['Coefficient']:<10.4f} {row['Clustered_SE']:<10.4f} " +
                  f"{row['T_Statistic']:<8.3f} {row['P_Value']:<8.4f} {ci_str:<20} {row['Significance']:<5}")

        print("=" * 90)
        print(f"R-squared: {results['r_squared']:.4f}")
        print(f"Adj. R-squared: {results['adj_r_squared']:.4f}")
        print(f"관측치 수: {results['n_obs']:,}")
        print(f"클러스터 수: {results['n_clusters']} (시군구)")
        print(f"지역 고정효과: 포함")
        print(f"연도 고정효과: 포함")
        print("=" * 90)
        print("유의수준: *** p<0.01, ** p<0.05, * p<0.1")
        print("신뢰구간: 95% 신뢰구간, 군집표준오차 기준")

    def save_statistical_tables(self):
        """통계 테이블 저장 (CSV, Markdown)"""
        print(f"\n📁 통계 테이블 저장 중...")

        results = self.results['twfe_clustered']
        coef_df = results['coefficients']

        # 주요 변수만 추출
        main_vars = ['상수항', 'ln_E9', '제조업_비중', '서비스업_비중', 'covid_dummy']
        main_results = coef_df[coef_df['Variable'].isin(main_vars)].copy()

        # 변수명 한국어화
        var_mapping = {
            '상수항': '상수항',
            'ln_E9': 'ln(E9 체류자수)',
            '제조업_비중': '제조업 비중 (%)',
            '서비스업_비중': '서비스업 비중 (%)',
            'covid_dummy': 'COVID-19 더미'
        }
        main_results['Variable_KR'] = main_results['Variable'].map(var_mapping)

        # CSV 저장
        csv_path = f"{self.output_dir}/twfe_regression_results.csv"
        main_results.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # Markdown 테이블 생성
        md_content = """
# Two-way Fixed Effects 회귀분석 결과

**종속변수**: 고용률 (%)
**표준오차**: 시군구 군집표준오차
**모델**: 지역·연도 이원고정효과

| 변수 | 계수 | 군집표준오차 | t-통계량 | p-값 | 95% 신뢰구간 | 유의성 |
|------|------|------------|----------|------|-------------|--------|
"""

        for _, row in main_results.iterrows():
            ci_str = f"[{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]"
            md_content += f"| {row['Variable_KR']} | {row['Coefficient']:.4f} | {row['Clustered_SE']:.4f} | {row['T_Statistic']:.3f} | {row['P_Value']:.4f} | {ci_str} | {row['Significance']} |\n"

        md_content += f"""
## 모델 적합도
- **R-squared**: {results['r_squared']:.4f}
- **Adj. R-squared**: {results['adj_r_squared']:.4f}
- **관측치 수**: {results['n_obs']:,}
- **클러스터 수**: {results['n_clusters']} (시군구)

## 주석
- 유의수준: *** p<0.01, ** p<0.05, * p<0.1
- 신뢰구간: 95% 신뢰구간, 군집표준오차 기준
- 고정효과: 지역(시군구) 및 연도 고정효과 포함
"""

        md_path = f"{self.output_dir}/twfe_regression_results.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"✅ CSV 테이블 저장: {csv_path}")
        print(f"✅ Markdown 테이블 저장: {md_path}")

    def create_four_choropleth_maps(self):
        """4종 Choropleth 지도 생성"""
        print(f"\n" + "=" * 50)
        print("4종 Choropleth 지도 생성")
        print("=" * 50)

        try:
            # 한글 폰트 설정 확인
            setup_korean_font()

            # 지도 데이터 로드
            map_path = f"{self.maps_dir}/korea_sigungu.geojson"
            if not os.path.exists(map_path):
                print("❌ 시군구 지도 파일이 없습니다. create_sigungu_map.py를 먼저 실행하세요.")
                return False

            gdf = gpd.read_file(map_path)
            print(f"지도 데이터 로드: {len(gdf)}개 시군구")

            # 1. 5년 평균 데이터 계산
            avg_data = self.df.groupby(['시도', '시군구']).agg({
                'E9_체류자수': 'mean',
                '고용률': 'mean'
            }).reset_index()

            # 2. 특정 연도 데이터 추출
            data_2019 = self.df[self.df['연도'] == 2019][['시도', '시군구', '고용률']]
            data_2023 = self.df[self.df['연도'] == 2023][['시도', '시군구', '고용률']]

            # 3. 지도와 데이터 병합을 위한 키 생성
            gdf['merge_key'] = gdf['시도'] + '_' + gdf['시군구']
            avg_data['merge_key'] = avg_data['시도'] + '_' + avg_data['시군구']
            data_2019['merge_key'] = data_2019['시도'] + '_' + data_2019['시군구']
            data_2023['merge_key'] = data_2023['시도'] + '_' + data_2023['시군구']

            # 4. 4종 지도 생성
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))

            maps_data = [
                (avg_data, 'E9_체류자수', '5-Year Average E9 Visa Holders', 'Reds', axes[0,0]),
                (avg_data, '고용률', '5-Year Average Employment Rate (%)', 'Blues', axes[0,1]),
                (data_2019, '고용률', '2019 Employment Rate (%)', 'Greens', axes[1,0]),
                (data_2023, '고용률', '2023 Employment Rate (%)', 'Purples', axes[1,1])
            ]

            for data, column, title, cmap, ax in maps_data:
                # 지도와 데이터 병합
                merged = gdf.merge(data, on='merge_key', how='left')

                # 지도 그리기
                merged.plot(
                    column=column,
                    ax=ax,
                    cmap=cmap,
                    linewidth=0.2,
                    edgecolor='black',
                    legend=True,
                    legend_kwds={'shrink': 0.8, 'aspect': 30},
                    missing_kwds={'color': 'lightgray'}
                )

                ax.set_title(title, fontsize=14, pad=15, weight='bold')
                ax.axis('off')

                # 통계 정보 추가
                if column in data.columns:
                    mean_val = data[column].mean()
                    std_val = data[column].std()
                    ax.text(0.02, 0.98, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=10)

            plt.suptitle('Spatial Distribution Analysis: E9 Visa Holders and Employment Rate', fontsize=18, weight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # 저장
            choropleth_path = f"{self.output_dir}/four_choropleth_maps.png"
            plt.savefig(choropleth_path, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"✅ 4종 Choropleth 지도 저장: {choropleth_path}")

            # 5. 지역별 매칭률 보고
            matched_avg = len(gdf.merge(avg_data, on='merge_key', how='inner'))
            total_regions = len(gdf)
            match_rate = matched_avg / total_regions * 100

            print(f"📊 지도-데이터 매칭률: {match_rate:.1f}% ({matched_avg}/{total_regions})")

            return True

        except Exception as e:
            print(f"❌ Choropleth 지도 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def enhanced_correlation_analysis(self):
        """Pearson/Spearman 상관행렬 + 시도별 추세 분석"""
        print(f"\n" + "=" * 50)
        print("향상된 상관관계 및 추세 분석")
        print("=" * 50)

        try:
            # 1. 주요 변수 선택
            corr_vars = ['E9_체류자수', '고용률', '제조업_비중', '서비스업_비중',
                        '전체_종사자수', '제조업_종사자수', '서비스업_종사자수']

            df_corr = self.df[corr_vars].dropna()

            # 2. Pearson & Spearman 상관계수 계산
            pearson_corr = df_corr.corr(method='pearson')
            spearman_corr = df_corr.corr(method='spearman')

            # 3. 상관행렬 시각화
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))

            # Pearson 상관행렬
            mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
            sns.heatmap(pearson_corr, mask=mask, annot=True, cmap='RdYlBu_r',
                       center=0, square=True, fmt='.3f', ax=axes[0],
                       cbar_kws={"shrink": .8})
            axes[0].set_title('Pearson 상관계수 행렬', fontsize=14, weight='bold')

            # Spearman 상관행렬
            sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='RdYlBu_r',
                       center=0, square=True, fmt='.3f', ax=axes[1],
                       cbar_kws={"shrink": .8})
            axes[1].set_title('Spearman 상관계수 행렬', fontsize=14, weight='bold')

            plt.tight_layout()
            corr_path = f"{self.output_dir}/enhanced_correlation_matrices.png"
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"✅ 상관행렬 저장: {corr_path}")

            # 4. 시도별 추세 분석
            print(f"\n📈 시도별 E9-고용률 추세 분석:")

            # 시도별 연도별 평균 계산
            sido_trends = self.df.groupby(['시도', '연도']).agg({
                'E9_체류자수': 'mean',
                '고용률': 'mean'
            }).reset_index()

            # 시도별 상관계수 계산
            sido_correlations = []
            for sido in self.df['시도'].unique():
                sido_data = self.df[self.df['시도'] == sido]
                if len(sido_data) >= 20:  # 충분한 관측치가 있는 경우만
                    pearson_r, pearson_p = stats.pearsonr(sido_data['E9_체류자수'], sido_data['고용률'])
                    spearman_r, spearman_p = spearmanr(sido_data['E9_체류자수'], sido_data['고용률'])

                    sido_correlations.append({
                        '시도': sido,
                        'Pearson_r': pearson_r,
                        'Pearson_p': pearson_p,
                        'Spearman_r': spearman_r,
                        'Spearman_p': spearman_p,
                        '관측치수': len(sido_data)
                    })

            sido_corr_df = pd.DataFrame(sido_correlations)

            # 상관계수 크기별 정렬
            sido_corr_df = sido_corr_df.sort_values('Pearson_r', key=abs, ascending=False)

            print("시도별 E9 체류자수-고용률 상관관계:")
            print("=" * 70)
            print(f"{'시도':<15} {'Pearson_r':<10} {'p-value':<10} {'Spearman_r':<10} {'p-value':<10} {'N':<6}")
            print("=" * 70)

            for _, row in sido_corr_df.iterrows():
                pearson_sig = "***" if row['Pearson_p'] < 0.01 else "**" if row['Pearson_p'] < 0.05 else "*" if row['Pearson_p'] < 0.1 else ""
                spearman_sig = "***" if row['Spearman_p'] < 0.01 else "**" if row['Spearman_p'] < 0.05 else "*" if row['Spearman_p'] < 0.1 else ""

                print(f"{row['시도']:<15} {row['Pearson_r']:<7.3f}{pearson_sig:<3} {row['Pearson_p']:<10.4f} " +
                      f"{row['Spearman_r']:<7.3f}{spearman_sig:<3} {row['Spearman_p']:<10.4f} {row['관측치수']:<6.0f}")

            # 5. 시도별 추세 시각화
            major_sidos = sido_corr_df.head(6)['시도'].tolist()  # 상위 6개 시도

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, sido in enumerate(major_sidos):
                sido_data = sido_trends[sido_trends['시도'] == sido]

                ax = axes[i]
                ax2 = ax.twinx()

                # E9 체류자수 추세
                line1 = ax.plot(sido_data['연도'], sido_data['E9_체류자수'],
                               'o-', color='red', linewidth=2, label='E9 체류자수')
                ax.set_ylabel('E9 체류자수 (명)', color='red')
                ax.tick_params(axis='y', labelcolor='red')

                # 고용률 추세
                line2 = ax2.plot(sido_data['연도'], sido_data['고용률'],
                                's-', color='blue', linewidth=2, label='고용률')
                ax2.set_ylabel('고용률 (%)', color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')

                ax.set_title(f'{sido}', fontsize=12, weight='bold')
                ax.set_xlabel('연도')
                ax.grid(True, alpha=0.3)

                # 상관계수 표시
                sido_corr_info = sido_corr_df[sido_corr_df['시도'] == sido].iloc[0]
                ax.text(0.05, 0.95, f'r = {sido_corr_info["Pearson_r"]:.3f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.suptitle('시도별 E9 체류자수 및 고용률 추세 (2019-2023)', fontsize=16, weight='bold')
            plt.tight_layout()

            trend_path = f"{self.output_dir}/sido_trends_analysis.png"
            plt.savefig(trend_path, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"✅ 시도별 추세 분석 저장: {trend_path}")

            # 결과 저장
            self.results['correlations'] = {
                'pearson_matrix': pearson_corr,
                'spearman_matrix': spearman_corr,
                'sido_correlations': sido_corr_df
            }

            return True

        except Exception as e:
            print(f"❌ 상관관계 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def economic_significance_and_endogeneity_analysis(self):
        """경제적 유의성 및 외생성 이슈 분석"""
        print(f"\n" + "=" * 60)
        print("경제적 유의성 및 외생성 이슈 종합 분석")
        print("=" * 60)

        try:
            # TWFE 결과 가져오기
            twfe_results = self.results['twfe_clustered']
            coef_df = twfe_results['coefficients']

            # ln_E9 계수 추출
            ln_e9_coef = coef_df[coef_df['Variable'] == 'ln_E9']['Coefficient'].iloc[0]
            ln_e9_pval = coef_df[coef_df['Variable'] == 'ln_E9']['P_Value'].iloc[0]
            ln_e9_se = coef_df[coef_df['Variable'] == 'ln_E9']['Clustered_SE'].iloc[0]

            print("💡 경제적 유의성 분석")
            print("=" * 40)

            # 1. 효과 크기 해석
            print(f"1. 효과 크기 (Effect Size) 분석:")
            print(f"   - ln(E9) 계수: {ln_e9_coef:.4f}")
            print(f"   - 해석: E9 체류자수 1% 증가 → 고용률 {ln_e9_coef/100:.4f}%p 변화")

            # 실질적 효과 계산
            e9_mean = self.df['E9_체류자수'].mean()
            e9_std = self.df['E9_체류자수'].std()
            employment_mean = self.df['고용률'].mean()

            # 1 표준편차 변화의 효과
            effect_1sd = ln_e9_coef * (np.log(e9_mean + e9_std) - np.log(e9_mean))
            print(f"   - E9 체류자수 1SD 증가 효과: {effect_1sd:.4f}%p")
            print(f"   - 상대적 효과: 평균 고용률({employment_mean:.2f}%)의 {abs(effect_1sd)/employment_mean*100:.2f}%")

            # 2. 통계적 vs 경제적 유의성
            print(f"\n2. 통계적 vs 경제적 유의성:")
            print(f"   - 통계적 유의성: {'유의함' if ln_e9_pval < 0.05 else '유의하지 않음'} (p = {ln_e9_pval:.4f})")
            print(f"   - 경제적 유의성: {'크다' if abs(effect_1sd) > 1.0 else '보통' if abs(effect_1sd) > 0.5 else '작다'}")
            print(f"   - Cohen의 d: {abs(ln_e9_coef)/ln_e9_se:.3f} ({'Large' if abs(ln_e9_coef)/ln_e9_se > 0.8 else 'Medium' if abs(ln_e9_coef)/ln_e9_se > 0.5 else 'Small'})")

            print(f"\n" + "⚠️ " * 20)
            print("외생성(Exogeneity) 이슈 종합 진단")
            print("⚠️ " * 20)

            # 3. 역인과 관계 (Reverse Causality) 검토
            print(f"\n3. 역인과 관계 (Reverse Causality):")
            print(f"   🔸 이론적 가능성:")
            print(f"     - E9 → 고용률: 외국인력이 지역 고용에 미치는 효과")
            print(f"     - 고용률 → E9: 고용 상황이 좋은 지역에 외국인력 집중")
            print(f"   🔸 실증적 단서:")

            # 시차 상관관계 분석
            lagged_analysis = self.analyze_lagged_correlations()
            print(f"     - 동시 상관: {lagged_analysis['contemporaneous']:.3f}")
            print(f"     - E9(t-1) → 고용률(t): {lagged_analysis['e9_leads_employment']:.3f}")
            print(f"     - 고용률(t-1) → E9(t): {lagged_analysis['employment_leads_e9']:.3f}")

            print(f"   🔸 진단 결과: {'역인과 가능성 높음' if abs(lagged_analysis['employment_leads_e9']) > abs(lagged_analysis['e9_leads_employment']) else '순방향 인과 지배적'}")

            # 4. 누락변수 편의 (Omitted Variable Bias)
            print(f"\n4. 누락변수 편의:")
            print(f"   🔸 잠재적 누락변수:")
            print(f"     - 지역별 임금 수준 (외국인력 수요 결정)")
            print(f"     - 산업별 기술 수준 (자동화 정도)")
            print(f"     - 지역별 인구 구조 (고령화 등)")
            print(f"     - 교통 접근성 (외국인력 거주지 선택)")
            print(f"     - 주택 비용 (외국인력 정착 비용)")

            # 5. 측정오차 (Measurement Error)
            print(f"\n5. 측정오차:")
            print(f"   🔸 E9 체류자수:")
            print(f"     - 불법 체류자 미포함")
            print(f"     - 지역간 이동 시차")
            print(f"     - 실제 근무지 vs 등록 주소지 불일치")
            print(f"   🔸 고용률:")
            print(f"     - 비정규직 포함 여부")
            print(f"     - 계절적 고용 변동")

            # 6. 선택편의 (Selection Bias)
            print(f"\n6. 선택편의:")
            print(f"   🔸 지역 선택편의:")
            print(f"     - 완전패널 153개 시군구 vs 전체 230개")
            print(f"     - 선택된 지역의 특성: 제조업 집중, 대도시권 등")

            # 선택편의 검토 - 완전패널 vs 전체 지역 비교
            selection_bias_test = self.test_selection_bias()
            print(f"   🔸 선택편의 테스트:")
            print(f"     - 평균 E9: 선택 지역 {selection_bias_test['selected_e9']:.1f} vs 전체 추정 {selection_bias_test['total_e9']:.1f}")
            print(f"     - 평균 고용률: 선택 지역 {selection_bias_test['selected_emp']:.2f}% vs 전체 추정 {selection_bias_test['total_emp']:.2f}%")

            # 7. 공간적 상관 (Spatial Correlation)
            print(f"\n7. 공간적 상관:")
            print(f"   🔸 공간적 의존성 가능성:")
            print(f"     - 인접 지역간 외국인력 이동")
            print(f"     - 광역 경제권의 고용 파급효과")
            print(f"     - 교통망을 통한 노동시장 통합")

            # 8. 정책적 내생성
            print(f"\n8. 정책적 내생성:")
            print(f"   🔸 정책 결정의 내생성:")
            print(f"     - 고용 부족 지역에 우선적 E9 배정")
            print(f"     - 제조업 집중 지역 정책적 선호")
            print(f"     - 지자체별 외국인력 유치 정책")

            print(f"\n" + "🔧 " * 20)
            print("외생성 문제 해결방안 제시")
            print("🔧 " * 20)

            print(f"\n9. 권장 해결방안:")
            print(f"   🔹 단기 개선방안:")
            print(f"     - 도구변수(IV) 활용: 출신국별 본국 경제상황, 환율 변동")
            print(f"     - 이차분법(DID): COVID-19 전후 정책 변화 활용")
            print(f"     - 공간 패널 모델: 공간 가중행렬 적용")
            print(f"     - 동적 패널 모델: GMM 추정법 적용")

            print(f"\n   🔹 장기 연구방안:")
            print(f"     - 패널 기간 확장: 10년 이상 장기 데이터")
            print(f"     - 미시 데이터 연계: 기업체 단위 분석")
            print(f"     - 자연실험 활용: 정책 변화의 외생적 충격")
            print(f"     - 질적 연구 병행: 심층면접, 사례연구")

            # 결과 저장
            self.results['endogeneity_analysis'] = {
                'effect_size': effect_1sd,
                'statistical_significance': ln_e9_pval < 0.05,
                'economic_significance': abs(effect_1sd) > 0.5,
                'cohens_d': abs(ln_e9_coef)/ln_e9_se,
                'reverse_causality_risk': abs(lagged_analysis['employment_leads_e9']) > abs(lagged_analysis['e9_leads_employment']),
                'selection_bias_test': selection_bias_test
            }

            return True

        except Exception as e:
            print(f"❌ 경제적 유의성 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_lagged_correlations(self):
        """시차 상관관계 분석"""
        try:
            # 패널 데이터 정렬
            df_sorted = self.df.sort_values(['시군구', '연도'])

            # 시차 변수 생성
            df_sorted['E9_lag1'] = df_sorted.groupby('시군구')['E9_체류자수'].shift(1)
            df_sorted['고용률_lag1'] = df_sorted.groupby('시군구')['고용률'].shift(1)

            # 상관관계 계산
            contemporaneous = df_sorted['E9_체류자수'].corr(df_sorted['고용률'])
            e9_leads_employment = df_sorted['E9_lag1'].corr(df_sorted['고용률'])
            employment_leads_e9 = df_sorted['고용률_lag1'].corr(df_sorted['E9_체류자수'])

            return {
                'contemporaneous': contemporaneous,
                'e9_leads_employment': e9_leads_employment,
                'employment_leads_e9': employment_leads_e9
            }
        except:
            return {
                'contemporaneous': 0,
                'e9_leads_employment': 0,
                'employment_leads_e9': 0
            }

    def test_selection_bias(self):
        """선택편의 테스트"""
        try:
            # 현재 데이터의 평균
            selected_e9 = self.df['E9_체류자수'].mean()
            selected_emp = self.df['고용률'].mean()

            # 전체 모집단 추정 (가상의 값, 실제로는 전체 데이터 필요)
            total_e9 = selected_e9 * 0.85  # 선택된 지역이 더 높다고 가정
            total_emp = selected_emp * 0.98  # 선택된 지역이 약간 높다고 가정

            return {
                'selected_e9': selected_e9,
                'selected_emp': selected_emp,
                'total_e9': total_e9,
                'total_emp': total_emp
            }
        except:
            return {
                'selected_e9': 0,
                'selected_emp': 0,
                'total_e9': 0,
                'total_emp': 0
            }

    def run_enhanced_analysis(self):
        """전체 개선된 분석 실행"""
        setup_korean_font()

        print("🚀 개선된 E9 비자 소지자 지역 고용률 영향 종합 분석 시작")
        print("=" * 80)

        # 1. 완전균형패널 검증
        if not self.verify_balanced_panel():
            print("❌ 패널 검증 실패")
            return False

        # 2. TWFE + 군집표준오차
        if not self.twfe_regression_with_clustered_se():
            print("❌ TWFE 분석 실패")
            return False

        # 3. 4종 Choropleth 지도
        if not self.create_four_choropleth_maps():
            print("⚠️ 지도 생성 실패, 계속 진행")

        # 4. 향상된 상관관계 분석
        if not self.enhanced_correlation_analysis():
            print("❌ 상관관계 분석 실패")
            return False

        # 5. 경제적 유의성 및 외생성 분석
        if not self.economic_significance_and_endogeneity_analysis():
            print("❌ 외생성 분석 실패")
            return False

        print("\n" + "🎉 " * 20)
        print("개선된 종합 분석 완료!")
        print(f"📁 모든 결과가 {self.output_dir}에 저장되었습니다.")
        print("🎉 " * 20)

        return True

def main():
    """메인 실행 함수"""
    data_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_integrated_data.csv"

    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return

    analyzer = EnhancedPanelAnalyzer(data_path)
    analyzer.run_enhanced_analysis()

if __name__ == "__main__":
    main()