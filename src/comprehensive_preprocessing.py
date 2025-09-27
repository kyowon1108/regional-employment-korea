import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_existing_processed_data():
    """기존 전처리된 데이터들 로드"""
    print("=== 기존 전처리된 데이터 로드 ===")

    # 1. 표준 시도-시군구 매핑 테이블
    mapping_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/data/processed/시도_시군구_매핑.csv")
    print(f"매핑 테이블: {len(mapping_df)}개 지자체")

    # 2. 산업별 종사자 데이터
    industry_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/data/processed/산업별_종사자_전처리완료_utf-8.csv")
    print(f"산업별 종사자: {len(industry_df)}개 관측치")

    # 3. E9 체류자 데이터
    e9_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/data/processed/E9_체류자_전처리완료_utf-8.csv")
    print(f"E9 체류자: {len(e9_df)}개 관측치")

    # 4. 고용률 원본 데이터
    employment_raw = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/employment_raw.csv")
    print(f"고용률 원본: {len(employment_raw)}개 관측치")

    return mapping_df, industry_df, e9_df, employment_raw

def clean_employment_data(employment_raw, mapping_df):
    """고용률 데이터 정제"""
    print("\\n=== 고용률 데이터 정제 ===")

    # 매핑 딕셔너리 생성
    mapping_dict = dict(zip(mapping_df['시군구'], mapping_df['시도']))
    print(f"매핑 딕셔너리 생성: {len(mapping_dict)}개 지역")

    employment_clean = []

    # 각 행을 처리
    for _, row in employment_raw.iterrows():
        region = row['행정구역']

        # 시도와 시군구 분리
        sido, sigungu = split_region_name(region)

        # 매핑 테이블로 시도명 표준화
        if sigungu in mapping_dict:
            sido_standard = mapping_dict[sigungu]
        else:
            sido_standard = sido

        # 2019-2023년 데이터 처리 (반기별)
        years_semesters = [
            ('2019', '1', '2019.1/2'), ('2019', '2', '2019.2/2'),
            ('2020', '1', '2020.1/2'), ('2020', '2', '2020.2/2'),
            ('2021', '1', '2021.1/2'), ('2021', '2', '2021.2/2'),
            ('2022', '1', '2022.1/2'), ('2022', '2', '2022.2/2'),
            ('2023', '1', '2023.1/2'), ('2023', '2', '2023.2/2')
        ]

        for year, semester, col_name in years_semesters:
            employment_rate = row[col_name]

            # '-' 값이 아니고 유효한 숫자인 경우만 포함
            if employment_rate != '-' and pd.notna(employment_rate):
                try:
                    employment_rate = float(employment_rate)
                    employment_clean.append({
                        '시도': sido_standard,
                        '시군구': sigungu,
                        '연도': int(year),
                        '반기': int(semester),
                        '고용률': employment_rate
                    })
                except:
                    continue

    employment_df = pd.DataFrame(employment_clean)
    print(f"정제된 고용률 데이터: {len(employment_df)}개 관측치")

    # 연도별 데이터 현황
    print("연도별 고용률 데이터 현황:")
    print(employment_df.groupby('연도').size())

    return employment_df

def split_region_name(region_name):
    """지역명을 시도와 시군구로 분리"""
    region_name = str(region_name).strip()

    # 특별 처리가 필요한 케이스들
    special_cases = {
        '광주시': ('광주광역시', '광주광역시'),  # 광주광역시 전체 데이터
        '제주시': ('제주특별자치도', '제주시'),   # 제주시 개별 데이터
        '대구 군위군': ('대구광역시', '군위군'),  # 대구 군위군
        '제주도 서귀포시': ('제주특별자치도', '서귀포시'),  # 제주도 옛 명칭
        '제주도 제주시': ('제주특별자치도', '제주시')       # 제주도 옛 명칭
    }

    if region_name in special_cases:
        return special_cases[region_name]

    # 시도 패턴 정의
    sido_patterns = {
        '서울': '서울특별시',
        '부산': '부산광역시',
        '대구': '대구광역시',
        '인천': '인천광역시',
        '광주': '광주광역시',
        '대전': '대전광역시',
        '울산': '울산광역시',
        '세종': '세종특별자치시',
        '경기': '경기도',
        '강원': '강원특별자치도',
        '충북': '충청북도',
        '충남': '충청남도',
        '전북': '전북특별자치도',
        '전남': '전라남도',
        '경북': '경상북도',
        '경남': '경상남도',
        '제주': '제주특별자치도',
        '제주도': '제주특별자치도'  # 제주도 옛 명칭 처리
    }

    # 패턴 매칭
    for pattern, full_name in sido_patterns.items():
        if region_name.startswith(pattern):
            sido = full_name
            sigungu = region_name[len(pattern):].strip()
            return sido, sigungu

    # 매칭되지 않는 경우
    return '미확인', region_name

def convert_to_annual_data(df):
    """반기별 데이터를 연도별로 변환 (평균 계산)"""
    print(f"\\n=== 반기별 → 연도별 데이터 변환 ===")
    print(f"변환 전: {len(df)}개 관측치")

    # 수치형 컬럼들 확인
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # 그룹화할 컬럼 (시도, 시군구, 연도)
    group_columns = ['시도', '시군구', '연도']

    # 수치형 컬럼에서 그룹화 컬럼 제외
    agg_columns = [col for col in numeric_columns if col not in group_columns and col != '반기']

    if len(agg_columns) == 0:
        print("수치형 컬럼이 없어 변환을 건너뜁니다.")
        return df.drop('반기', axis=1) if '반기' in df.columns else df

    # 연도별 평균 계산
    agg_dict = {col: 'mean' for col in agg_columns}

    annual_df = df.groupby(group_columns).agg(agg_dict).reset_index()

    print(f"변환 후: {len(annual_df)}개 관측치")
    return annual_df

def standardize_region_names(df, mapping_dict):
    """지역명 표준화"""
    print("\\n지역명 표준화 중...")

    standardized_count = 0
    for idx, row in df.iterrows():
        sigungu = row['시군구']
        if sigungu in mapping_dict and (row['시도'] == '미확인' or pd.isna(row['시도'])):
            df.loc[idx, '시도'] = mapping_dict[sigungu]
            standardized_count += 1

    print(f"표준화된 지역: {standardized_count}개")
    return df

def process_special_regions(df):
    """특별 처리가 필요한 지역들 처리"""
    if df.empty:
        return df

    df_processed = df.copy()

    # 0. 제주도 → 제주특별자치도 변환
    jeju_old_data = df_processed[df_processed['시도'] == '제주도']
    if len(jeju_old_data) > 0:
        print("제주도 → 제주특별자치도 변환 중...")
        df_processed.loc[df_processed['시도'] == '제주도', '시도'] = '제주특별자치도'

    # 1. 광주광역시 구별 데이터 합산
    gwangju_gu_data = df_processed[
        (df_processed['시도'] == '광주광역시') &
        (df_processed['시군구'].isin(['동구', '서구', '남구', '북구', '광산구']))
    ]

    if len(gwangju_gu_data) > 0:
        print("광주광역시 구별 데이터 통합 중...")
        # 구별 데이터를 연도별로 합산
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        agg_cols = [col for col in numeric_cols if col not in ['연도']]

        gwangju_aggregated = gwangju_gu_data.groupby(['시도', '연도']).agg({
            col: 'sum' for col in agg_cols if col in gwangju_gu_data.columns
        }).reset_index()
        gwangju_aggregated['시군구'] = '광주광역시'

        # 원래 구별 데이터 제거하고 통합 데이터 추가
        df_processed = df_processed[~(
            (df_processed['시도'] == '광주광역시') &
            (df_processed['시군구'].isin(['동구', '서구', '남구', '북구', '광산구']))
        )]
        df_processed = pd.concat([df_processed, gwangju_aggregated], ignore_index=True)

    # 2. 대구 군위군 편입 처리 (경상북도 군위군 + 대구광역시 군위군)
    gyeongbuk_gunwi = df_processed[
        (df_processed['시도'] == '경상북도') & (df_processed['시군구'] == '군위군')
    ]
    daegu_gunwi = df_processed[
        (df_processed['시도'] == '대구광역시') & (df_processed['시군구'] == '군위군')
    ]

    if len(gyeongbuk_gunwi) > 0 or len(daegu_gunwi) > 0:
        print("대구 군위군 편입 데이터 통합 중...")
        combined_gunwi = []
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        agg_cols = [col for col in numeric_cols if col not in ['연도']]

        for year in df_processed['연도'].unique():
            gb_year = gyeongbuk_gunwi[gyeongbuk_gunwi['연도'] == year]
            dg_year = daegu_gunwi[daegu_gunwi['연도'] == year]

            if len(gb_year) > 0 and len(dg_year) > 0:
                # 두 데이터 합산
                combined_row = gb_year.iloc[0].copy()
                combined_row['시도'] = '대구광역시'
                for col in agg_cols:
                    if col in gb_year.columns and col in dg_year.columns:
                        combined_row[col] = gb_year[col].iloc[0] + dg_year[col].iloc[0]
                combined_gunwi.append(combined_row)
            elif len(dg_year) > 0:
                # 대구 데이터만 있는 경우
                combined_gunwi.append(dg_year.iloc[0])
            elif len(gb_year) > 0:
                # 경북 데이터만 있는 경우 (대구로 변경)
                combined_row = gb_year.iloc[0].copy()
                combined_row['시도'] = '대구광역시'
                combined_gunwi.append(combined_row)

        if combined_gunwi:
            # 기존 군위군 데이터 제거하고 통합 데이터 추가
            df_processed = df_processed[~(
                ((df_processed['시도'] == '경상북도') | (df_processed['시도'] == '대구광역시')) &
                (df_processed['시군구'] == '군위군')
            )]
            combined_df = pd.DataFrame(combined_gunwi)
            df_processed = pd.concat([df_processed, combined_df], ignore_index=True)

    return df_processed

def integrate_all_datasets(mapping_df, industry_df, e9_df, employment_df):
    """모든 데이터셋 통합"""
    print("\\n=== 데이터셋 통합 ===")

    # 매핑 딕셔너리
    mapping_dict = dict(zip(mapping_df['시군구'], mapping_df['시도']))

    # 1. 각 데이터를 연도별로 변환
    print("1. 산업별 종사자 데이터 연도별 변환")
    industry_annual = convert_to_annual_data(industry_df)
    industry_annual = standardize_region_names(industry_annual, mapping_dict)
    industry_annual = process_special_regions(industry_annual)

    print("2. E9 체류자 데이터 연도별 변환")
    e9_annual = convert_to_annual_data(e9_df)
    e9_annual = standardize_region_names(e9_annual, mapping_dict)
    e9_annual = process_special_regions(e9_annual)

    print("3. 고용률 데이터 연도별 변환")
    employment_annual = convert_to_annual_data(employment_df)
    employment_annual = standardize_region_names(employment_annual, mapping_dict)

    # 2. 기준 데이터프레임 생성 (고용률 기준)
    print("\\n기준 데이터프레임 생성...")
    base_df = employment_annual.copy()
    print(f"기준 데이터: {len(base_df)}개 관측치")

    # 3. 산업별 종사자 데이터 병합
    print("산업별 종사자 데이터 병합...")
    merged_df = base_df.merge(
        industry_annual[['시도', '시군구', '연도', '제조업_종사자수', '서비스업_종사자수']],
        on=['시도', '시군구', '연도'],
        how='left'
    )
    print(f"병합 후: {len(merged_df)}개 관측치")

    # 4. E9 체류자 데이터 병합
    print("E9 체류자 데이터 병합...")
    merged_df = merged_df.merge(
        e9_annual[['시도', '시군구', '연도', 'E9_체류자수']],
        on=['시도', '시군구', '연도'],
        how='left'
    )
    print(f"병합 후: {len(merged_df)}개 관측치")

    # 5. 면적 데이터 병합
    print("면적 데이터 병합...")
    merged_df = merged_df.merge(
        mapping_df[['시도', '시군구', '면적']],
        on=['시도', '시군구'],
        how='left'
    )
    print(f"최종 병합: {len(merged_df)}개 관측치")

    # 6. 결측값 처리
    print("\\n결측값 처리...")
    merged_df['E9_체류자수'] = merged_df['E9_체류자수'].fillna(0)
    merged_df['제조업_종사자수'] = merged_df['제조업_종사자수'].fillna(0)
    merged_df['서비스업_종사자수'] = merged_df['서비스업_종사자수'].fillna(0)
    merged_df['면적'] = merged_df['면적'].fillna(0)

    # 7. 추가 변수 계산
    print("추가 변수 계산...")
    merged_df['전체_종사자수'] = merged_df['제조업_종사자수'] + merged_df['서비스업_종사자수']
    merged_df['제조업_비중'] = np.where(
        merged_df['전체_종사자수'] > 0,
        merged_df['제조업_종사자수'] / merged_df['전체_종사자수'] * 100,
        0
    )
    merged_df['서비스업_비중'] = np.where(
        merged_df['전체_종사자수'] > 0,
        merged_df['서비스업_종사자수'] / merged_df['전체_종사자수'] * 100,
        0
    )

    # 8. 컬럼 순서 정리
    final_columns = [
        '시도', '시군구', '면적', '연도',
        'E9_체류자수', '제조업_종사자수', '서비스업_종사자수', '전체_종사자수',
        '고용률', '제조업_비중', '서비스업_비중'
    ]

    # 존재하는 컬럼만 선택
    existing_columns = [col for col in final_columns if col in merged_df.columns]
    merged_df = merged_df[existing_columns]

    return merged_df

def filter_complete_data(integrated_df, required_years=[2019, 2020, 2021, 2022, 2023]):
    """완전한 데이터를 가진 지자체만 필터링"""
    print(f"\\n=== 완전한 데이터 필터링 ({required_years}) ===")

    complete_municipalities = []

    for (sido, sigungu), group in integrated_df.groupby(['시도', '시군구']):
        # 유효한 고용률 데이터가 있는 연도들 확인
        valid_years = set(group[group['고용률'] > 0]['연도'].unique())

        # 요구되는 모든 연도 데이터가 있는지 확인
        if set(required_years).issubset(valid_years):
            complete_municipalities.append((sido, sigungu))

    print(f"완전한 데이터를 가진 지자체: {len(complete_municipalities)}개")

    if len(complete_municipalities) == 0:
        print("완전한 데이터를 가진 지자체가 없습니다. 2021-2023년으로 재시도...")
        return filter_complete_data(integrated_df, [2021, 2022, 2023])

    # 완전한 데이터를 가진 지자체만 필터링
    mask = integrated_df.apply(
        lambda row: (row['시도'], row['시군구']) in complete_municipalities,
        axis=1
    )

    final_df = integrated_df[mask].copy()
    print(f"최종 필터링된 데이터: {len(final_df)}개 관측치")

    # 시도별 지자체 수
    print("\\n시도별 지자체 수:")
    sido_counts = final_df.groupby('시도')['시군구'].nunique().sort_values(ascending=False)
    print(sido_counts)

    return final_df

def quality_check(df):
    """데이터 품질 확인"""
    print("\\n=== 데이터 품질 확인 ===")

    print(f"최종 데이터 크기: {df.shape}")
    print(f"지자체 수: {df[['시도', '시군구']].drop_duplicates().shape[0]}개")
    print(f"분석 기간: {df['연도'].min()}-{df['연도'].max()}년")

    # 결측값 확인
    print("\\n결측값 현황:")
    missing_info = df.isnull().sum()
    for col, count in missing_info.items():
        if count > 0:
            print(f"- {col}: {count}개 ({count/len(df)*100:.1f}%)")

    # 기본 통계
    print("\\n주요 변수 기본 통계:")
    key_vars = ['고용률', 'E9_체류자수', '제조업_비중', '서비스업_비중']
    for var in key_vars:
        if var in df.columns:
            print(f"- {var}: 평균 {df[var].mean():.2f}, 범위 {df[var].min():.2f}~{df[var].max():.2f}")

    # 지역명 표준화 상태
    print(f"\\n지역명 표준화 상태:")
    print(f"- 표준화된 시도 수: {df['시도'].nunique()}개")
    unmatched = len(df[df['시도'] == '미확인'])
    if unmatched > 0:
        print(f"- 미확인 지역: {unmatched}개 ({unmatched/len(df)*100:.1f}%)")
    else:
        print("- 모든 지역이 표준화되었습니다.")

    return df

def main():
    """메인 전처리 실행 함수"""
    print("=== 종합 데이터 전처리 시작 ===\\n")

    # 1. 기존 전처리된 데이터 로드
    mapping_df, industry_df, e9_df, employment_raw = load_existing_processed_data()

    # 2. 고용률 데이터 정제
    employment_clean = clean_employment_data(employment_raw, mapping_df)

    # 3. 모든 데이터셋 통합
    integrated_df = integrate_all_datasets(mapping_df, industry_df, e9_df, employment_clean)

    # 4. 완전한 데이터만 필터링
    final_df = filter_complete_data(integrated_df)

    # 5. 품질 확인
    final_df = quality_check(final_df)

    # 6. 저장
    output_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_integrated_data.csv"
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\\n최종 통합 데이터 저장: {output_path}")

    # 7. 요약 통계 생성
    summary_df = final_df.groupby(['시도', '시군구']).agg({
        '고용률': 'mean',
        'E9_체류자수': 'mean',
        '제조업_종사자수': 'mean',
        '서비스업_종사자수': 'mean',
        '전체_종사자수': 'mean',
        '면적': 'first',
        '제조업_비중': 'mean',
        '서비스업_비중': 'mean'
    }).reset_index()

    summary_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    print(f"지역별 요약 통계 저장: {summary_path}")

    print("\\n=== 종합 전처리 완료 ===")
    return final_df, summary_df

if __name__ == "__main__":
    final_df, summary_df = main()