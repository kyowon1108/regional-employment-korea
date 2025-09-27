import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_mapping_table():
    """표준 시도-시군구 매핑 테이블 로드"""
    mapping_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/data/processed/시도_시군구_매핑.csv")
    print(f"매핑 테이블 로드 완료: {len(mapping_df)}개 지자체")

    # 매핑 딕셔너리 생성 (시군구명 -> 시도)
    mapping_dict = dict(zip(mapping_df['시군구'], mapping_df['시도']))

    print("시도별 지자체 수:")
    print(mapping_df['시도'].value_counts().head(10))

    return mapping_df, mapping_dict

def standardize_region_names(df, mapping_dict):
    """지역명 표준화"""
    print(f"\n=== 지역명 표준화 ===")
    print(f"표준화 전 데이터: {len(df)}개 관측치")

    # 현재 시도 분포 확인
    print("\n표준화 전 시도 분포:")
    print(df['시도'].value_counts().head(10))

    # '미확인' 시도를 매핑 테이블로 수정
    updated_count = 0
    for idx, row in df.iterrows():
        if row['시도'] == '미확인' or pd.isna(row['시도']):
            sigungu = row['시군구']
            if sigungu in mapping_dict:
                df.loc[idx, '시도'] = mapping_dict[sigungu]
                updated_count += 1

    print(f"\n업데이트된 지역 수: {updated_count}개")

    # 표준화 후 시도 분포 확인
    print("\n표준화 후 시도 분포:")
    print(df['시도'].value_counts())

    # 여전히 '미확인'인 지역들 확인
    unmatched = df[df['시도'] == '미확인']
    if len(unmatched) > 0:
        print(f"\n여전히 매칭되지 않은 지역: {len(unmatched)}개")
        print(unmatched[['시도', '시군구']].drop_duplicates().head(10))

    return df

def update_integrated_data():
    """통합 데이터 업데이트"""
    print("=== 통합 데이터 업데이트 ===")

    # 매핑 테이블 로드
    mapping_df, mapping_dict = load_mapping_table()

    # 기존 통합 데이터 로드
    integrated_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/integrated_analysis_data.csv"
    df = pd.read_csv(integrated_path)

    print(f"기존 통합 데이터 로드: {len(df)}개 관측치")

    # 지역명 표준화
    df_standardized = standardize_region_names(df.copy(), mapping_dict)

    # 표준화된 데이터에서 완전한 2019-2023년 데이터를 가진 지자체만 필터링
    complete_municipalities = []

    for (sido, sigungu), group in df_standardized.groupby(['시도', '시군구']):
        # 2019-2023년 모든 연도에 고용률 데이터가 있는지 확인
        years_with_data = set()
        for _, row in group.iterrows():
            if pd.notna(row['고용률']) and row['고용률'] > 0:
                years_with_data.add(row['연도'])

        # 5개년 모든 데이터가 있는 경우
        if years_with_data == {2019, 2020, 2021, 2022, 2023}:
            complete_municipalities.append((sido, sigungu))

    print(f"\n2019-2023년 완전 데이터 보유 지자체: {len(complete_municipalities)}개")

    # 완전한 데이터를 가진 지자체만 필터링
    if len(complete_municipalities) > 0:
        mask = df_standardized.apply(
            lambda row: (row['시도'], row['시군구']) in complete_municipalities,
            axis=1
        )
        final_df = df_standardized[mask].copy()

        print(f"최종 필터링된 데이터: {len(final_df)}개 관측치")

        # 시도별 지자체 수 확인
        print("\n시도별 지자체 수 (최종):")
        final_counts = final_df.groupby('시도')['시군구'].nunique().sort_values(ascending=False)
        print(final_counts)

        # 결측값 처리
        final_df['E9_체류자수'] = final_df['E9_체류자수'].fillna(0)
        final_df['제조업_종사자수'] = final_df['제조업_종사자수'].fillna(0)
        final_df['전체_종사자수'] = final_df['전체_종사자수'].fillna(0)
        final_df['서비스업_종사자수'] = final_df['전체_종사자수'] - final_df['제조업_종사자수']
        final_df['서비스업_종사자수'] = final_df['서비스업_종사자수'].clip(lower=0)

        # 면적 정보 업데이트 (매핑 테이블의 정확한 면적으로)
        area_mapping = dict(zip(mapping_df['시군구'], mapping_df['면적']))
        for idx, row in final_df.iterrows():
            sigungu = row['시군구']
            if sigungu in area_mapping:
                final_df.loc[idx, '면적'] = area_mapping[sigungu]

        # 저장
        output_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/standardized_integrated_data.csv"
        final_df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"\n표준화된 통합 데이터 저장 완료: {output_path}")

        return final_df
    else:
        print("완전한 데이터를 가진 지자체가 없습니다.")
        return None

def create_region_summary():
    """지역별 요약 통계 생성"""
    print("\n=== 지역별 요약 통계 생성 ===")

    # 표준화된 데이터 로드
    df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/standardized_integrated_data.csv")

    # 5년 평균 계산
    summary_df = df.groupby(['시도', '시군구']).agg({
        '고용률': 'mean',
        'E9_체류자수': 'mean',
        '제조업_종사자수': 'mean',
        '서비스업_종사자수': 'mean',
        '전체_종사자수': 'mean',
        '면적': 'first'
    }).reset_index()

    # 추가 지표 계산
    summary_df['제조업_비중'] = (summary_df['제조업_종사자수'] / summary_df['전체_종사자수'] * 100).fillna(0)
    summary_df['서비스업_비중'] = (summary_df['서비스업_종사자수'] / summary_df['전체_종사자수'] * 100).fillna(0)
    summary_df['종사자_밀도'] = (summary_df['전체_종사자수'] / summary_df['면적']).replace([np.inf, -np.inf], 0).fillna(0)
    summary_df['E9_밀도'] = (summary_df['E9_체류자수'] / summary_df['면적']).replace([np.inf, -np.inf], 0).fillna(0)

    print(f"지역별 요약 통계: {len(summary_df)}개 지자체")

    # 시도별 요약
    sido_summary = summary_df.groupby('시도').agg({
        '고용률': ['mean', 'std', 'min', 'max'],
        'E9_체류자수': ['mean', 'sum'],
        '제조업_비중': 'mean',
        '서비스업_비중': 'mean',
        '시군구': 'count'
    }).round(2)

    sido_summary.columns = ['고용률_평균', '고용률_표준편차', '고용률_최소', '고용률_최대',
                           'E9_평균', 'E9_총합', '제조업비중_평균', '서비스업비중_평균', '지자체수']

    print("\n시도별 요약 통계:")
    print(sido_summary)

    # 저장
    summary_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/standardized_regional_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')

    sido_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/result_data/sido_level_summary.csv"
    sido_summary.to_csv(sido_path, encoding='utf-8')

    print(f"\n요약 통계 저장 완료:")
    print(f"- 지역별 요약: {summary_path}")
    print(f"- 시도별 요약: {sido_path}")

    return summary_df, sido_summary

def validate_data_quality():
    """데이터 품질 검증"""
    print("\n=== 데이터 품질 검증 ===")

    # 표준화된 데이터 로드
    df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/standardized_integrated_data.csv")

    print(f"최종 데이터 크기: {df.shape}")
    print(f"지자체 수: {df[['시도', '시군구']].drop_duplicates().shape[0]}개")
    print(f"분석 기간: {df['연도'].min()}-{df['연도'].max()}년")

    # 결측값 확인
    print("\n결측값 현황:")
    missing_count = df.isnull().sum()
    for col, count in missing_count.items():
        if count > 0:
            print(f"- {col}: {count}개 ({count/len(df)*100:.1f}%)")

    # 데이터 완전성 확인
    print("\n데이터 완전성 확인:")
    expected_records_per_region = 10  # 5년 × 2반기

    completeness_check = df.groupby(['시도', '시군구']).size()
    incomplete_regions = completeness_check[completeness_check != expected_records_per_region]

    if len(incomplete_regions) > 0:
        print(f"불완전한 데이터를 가진 지역: {len(incomplete_regions)}개")
        print(incomplete_regions.head())
    else:
        print("모든 지역이 완전한 10개 관측치를 보유하고 있습니다.")

    # 시도명 표준화 확인
    print(f"\n시도 표준화 현황:")
    print(f"- 표준 시도명 수: {df['시도'].nunique()}개")
    print(f"- '미확인' 지역: {len(df[df['시도'] == '미확인'])}개")

    # 고용률 범위 확인
    employment_stats = df['고용률'].describe()
    print(f"\n고용률 통계:")
    print(f"- 평균: {employment_stats['mean']:.2f}%")
    print(f"- 범위: {employment_stats['min']:.2f}% ~ {employment_stats['max']:.2f}%")
    print(f"- 표준편차: {employment_stats['std']:.2f}%")

    # E9 체류자수 분포 확인
    e9_stats = df['E9_체류자수'].describe()
    print(f"\nE9 체류자수 통계:")
    print(f"- 평균: {e9_stats['mean']:.1f}명")
    print(f"- 최대: {e9_stats['max']:.0f}명")
    print(f"- 0명인 관측치: {len(df[df['E9_체류자수'] == 0])}개 ({len(df[df['E9_체류자수'] == 0])/len(df)*100:.1f}%)")

    return df

def main():
    """메인 실행 함수"""
    print("데이터 표준화 시작...\n")

    # 1. 통합 데이터 업데이트 (지역명 표준화)
    standardized_df = update_integrated_data()

    if standardized_df is not None:
        # 2. 지역별 요약 통계 생성
        summary_df, sido_summary = create_region_summary()

        # 3. 데이터 품질 검증
        final_df = validate_data_quality()

        print("\n=== 데이터 표준화 완료 ===")
        print("생성된 파일:")
        print("- standardized_integrated_data.csv: 표준화된 통합 데이터")
        print("- standardized_regional_summary.csv: 지역별 요약 통계")
        print("- sido_level_summary.csv: 시도별 요약 통계")

        return final_df, summary_df, sido_summary
    else:
        print("데이터 표준화 실패")
        return None, None, None

if __name__ == "__main__":
    final_df, summary_df, sido_summary = main()