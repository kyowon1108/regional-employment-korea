import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def process_employment_data():
    """고용률 데이터 정제 및 변환"""
    print("=== 고용률 데이터 정제 ===")

    df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/employment_raw.csv")

    # 2019-2023년 컬럼만 선택
    employment_cols = ['2019.1/2', '2019.2/2', '2020.1/2', '2020.2/2', '2021.1/2', '2021.2/2', '2022.1/2', '2022.2/2', '2023.1/2', '2023.2/2']

    # 지역명과 고용률 데이터만 추출
    df_clean = df[['행정구역'] + employment_cols].copy()

    # 지역명 정제
    df_clean['행정구역_정제'] = df_clean['행정구역'].str.strip()

    # 시도와 시군구 분리
    sido_list = []
    sigungu_list = []

    for region in df_clean['행정구역_정제']:
        if pd.isna(region):
            sido_list.append('')
            sigungu_list.append('')
            continue

        region = str(region).strip()

        # 시도 패턴 매칭
        if region.startswith('서울'):
            sido = '서울특별시'
            sigungu = region.replace('서울 ', '').strip()
        elif region.startswith('부산'):
            sido = '부산광역시'
            sigungu = region.replace('부산 ', '').strip()
        elif region.startswith('대구'):
            sido = '대구광역시'
            sigungu = region.replace('대구 ', '').strip()
        elif region.startswith('인천'):
            sido = '인천광역시'
            sigungu = region.replace('인천 ', '').strip()
        elif region.startswith('광주'):
            sido = '광주광역시'
            sigungu = region.replace('광주 ', '').strip()
        elif region.startswith('대전'):
            sido = '대전광역시'
            sigungu = region.replace('대전 ', '').strip()
        elif region.startswith('울산'):
            sido = '울산광역시'
            sigungu = region.replace('울산 ', '').strip()
        elif region.startswith('세종'):
            sido = '세종특별자치시'
            sigungu = region.replace('세종 ', '').strip()
        elif region.startswith('경기'):
            sido = '경기도'
            sigungu = region.replace('경기 ', '').strip()
        elif region.startswith('강원'):
            sido = '강원특별자치도'
            sigungu = region.replace('강원 ', '').strip()
        elif region.startswith('충북'):
            sido = '충청북도'
            sigungu = region.replace('충북 ', '').strip()
        elif region.startswith('충남'):
            sido = '충청남도'
            sigungu = region.replace('충남 ', '').strip()
        elif region.startswith('전북'):
            sido = '전북특별자치도'
            sigungu = region.replace('전북 ', '').strip()
        elif region.startswith('전남'):
            sido = '전라남도'
            sigungu = region.replace('전남 ', '').strip()
        elif region.startswith('경북'):
            sido = '경상북도'
            sigungu = region.replace('경북 ', '').strip()
        elif region.startswith('경남'):
            sido = '경상남도'
            sigungu = region.replace('경남 ', '').strip()
        elif region.startswith('제주'):
            sido = '제주특별자치도'
            sigungu = region.replace('제주 ', '').strip()
        else:
            # 시도가 붙어있지 않은 경우 - 시군구명으로 추정
            sido = '미확인'
            sigungu = region

        sido_list.append(sido)
        sigungu_list.append(sigungu)

    df_clean['시도'] = sido_list
    df_clean['시군구'] = sigungu_list

    # Long format으로 변환
    employment_long = []
    for idx, row in df_clean.iterrows():
        sido = row['시도']
        sigungu = row['시군구']

        for col in employment_cols:
            year = int(col.split('.')[0])
            semester = int(col.split('/')[1])
            employment_rate = row[col]

            # '-' 값은 NaN으로 처리
            if employment_rate == '-' or pd.isna(employment_rate):
                employment_rate = np.nan
            else:
                try:
                    employment_rate = float(employment_rate)
                except:
                    employment_rate = np.nan

            employment_long.append({
                '시도': sido,
                '시군구': sigungu,
                '연도': year,
                '반기': semester,
                '고용률': employment_rate
            })

    employment_df = pd.DataFrame(employment_long)

    print(f"고용률 데이터 변환 완료: {len(employment_df)}개 관측치")
    print("연도별 데이터 분포:")
    print(employment_df.groupby('연도')['고용률'].count())

    return employment_df

def process_e9_data():
    """E9 체류자 데이터 정제 및 변환"""
    print("\\n=== E9 체류자 데이터 정제 ===")

    df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/e9_raw.csv")

    # 2019-2023년 컬럼만 선택
    year_cols = ['2019', '2020', '2021', '2022', '2023']

    # 지역별 데이터만 추출 (총계, 소계 제외)
    df_clean = df[
        (~df['행정구역(시군구)별(1)'].str.contains('총계', na=False)) &
        (~df['행정구역(시군구)별(2)'].str.contains('소계', na=False))
    ].copy()

    print(f"지역별 E9 데이터: {len(df_clean)}개 지역")

    # Long format으로 변환
    e9_long = []
    for idx, row in df_clean.iterrows():
        sido = str(row['행정구역(시군구)별(1)']).strip()
        sigungu = str(row['행정구역(시군구)별(2)']).strip()

        for year_col in year_cols:
            year = int(year_col)
            e9_count = row[year_col]

            # 숫자가 아닌 값 처리
            try:
                e9_count = int(e9_count) if pd.notna(e9_count) else 0
            except:
                e9_count = 0

            # 각 연도에 대해 1반기, 2반기 동일한 값으로 처리
            for semester in [1, 2]:
                e9_long.append({
                    '시도': sido,
                    '시군구': sigungu,
                    '연도': year,
                    '반기': semester,
                    'E9_체류자수': e9_count
                })

    e9_df = pd.DataFrame(e9_long)

    print(f"E9 데이터 변환 완료: {len(e9_df)}개 관측치")
    print("연도별 E9 체류자 합계:")
    print(e9_df.groupby('연도')['E9_체류자수'].sum())

    return e9_df

def process_industry_data():
    """산업별 종사자 데이터 정제"""
    print("\\n=== 산업별 종사자 데이터 정제 ===")

    df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/industry_raw.csv")

    # 제조업과 서비스업 종사자 수 추출
    manufacturing_df = df[
        (df['산업별'] == '광업.제조업(B,C)') &
        (df['항목'] == '전체종사자')
    ].copy()

    service_df = df[
        (df['산업별'].str.contains('서비스업', na=False)) &
        (df['항목'] == '전체종사자')
    ].copy()

    # 전체 종사자 수 추출
    total_df = df[
        (df['산업별'] == '전산업') &
        (df['항목'] == '전체종사자')
    ].copy()

    print(f"제조업 데이터: {len(manufacturing_df)}개")
    print(f"서비스업 데이터: {len(service_df)}개")
    print(f"전체 종사자 데이터: {len(total_df)}개")

    # 지역명 정제 및 변환
    def process_industry_long(df, industry_type):
        industry_long = []
        year_cols = ['2019. 1/2', '2019. 2/2', '2020. 1/2', '2020. 2/2',
                     '2021. 1/2', '2021. 2/2', '2022. 1/2', '2022. 2/2',
                     '2023. 1/2', '2023. 2/2']

        for idx, row in df.iterrows():
            region = str(row['지역별']).strip()

            # 시도와 시군구 분리
            if ' ' in region:
                parts = region.split(' ', 1)
                if len(parts) == 2:
                    sido_short, sigungu = parts
                    # 시도명 확장
                    sido_mapping = {
                        '서울특별시': '서울특별시',
                        '부산광역시': '부산광역시',
                        '대구광역시': '대구광역시',
                        '인천광역시': '인천광역시',
                        '광주광역시': '광주광역시',
                        '대전광역시': '대전광역시',
                        '울산광역시': '울산광역시',
                        '세종특별자치시': '세종특별자치시',
                        '경기도': '경기도',
                        '강원특별자치도': '강원특별자치도',
                        '충청북도': '충청북도',
                        '충청남도': '충청남도',
                        '전북특별자치도': '전북특별자치도',
                        '전라남도': '전라남도',
                        '경상북도': '경상북도',
                        '경상남도': '경상남도',
                        '제주특별자치도': '제주특별자치도'
                    }
                    sido = sido_mapping.get(sido_short, sido_short)
                else:
                    sido = '미확인'
                    sigungu = region
            else:
                sido = '미확인'
                sigungu = region

            for col in year_cols:
                try:
                    year = int(col.split('.')[0].strip())
                    semester = int(col.split('/')[1].strip())
                    value = row[col]

                    # 숫자가 아닌 값 처리
                    try:
                        value = int(value) if pd.notna(value) else 0
                    except:
                        value = 0

                    industry_long.append({
                        '시도': sido,
                        '시군구': sigungu,
                        '연도': year,
                        '반기': semester,
                        f'{industry_type}_종사자수': value
                    })
                except:
                    continue

        return pd.DataFrame(industry_long)

    manufacturing_long = process_industry_long(manufacturing_df, '제조업')
    total_long = process_industry_long(total_df, '전체')

    print(f"제조업 종사자 데이터 변환 완료: {len(manufacturing_long)}개 관측치")
    print(f"전체 종사자 데이터 변환 완료: {len(total_long)}개 관측치")

    return manufacturing_long, total_long

def integrate_all_data():
    """모든 데이터 통합"""
    print("\\n=== 데이터 통합 ===")

    # 각 데이터 처리
    employment_df = process_employment_data()
    e9_df = process_e9_data()
    manufacturing_df, total_df = process_industry_data()

    # 면적 데이터 로드
    area_df = pd.read_csv("/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/area_raw.csv")

    # 통합 데이터프레임 생성
    print("\\n데이터 병합 시작...")

    # 고용률 데이터를 기준으로 병합
    integrated_df = employment_df.copy()

    # E9 데이터 병합
    integrated_df = integrated_df.merge(
        e9_df[['시도', '시군구', '연도', '반기', 'E9_체류자수']],
        on=['시도', '시군구', '연도', '반기'],
        how='left'
    )

    # 제조업 종사자 데이터 병합
    integrated_df = integrated_df.merge(
        manufacturing_df[['시도', '시군구', '연도', '반기', '제조업_종사자수']],
        on=['시도', '시군구', '연도', '반기'],
        how='left'
    )

    # 전체 종사자 데이터 병합
    integrated_df = integrated_df.merge(
        total_df[['시도', '시군구', '연도', '반기', '전체_종사자수']],
        on=['시도', '시군구', '연도', '반기'],
        how='left'
    )

    # 면적 데이터 병합
    integrated_df = integrated_df.merge(
        area_df[['시도', '시군구', '면적']],
        on=['시도', '시군구'],
        how='left'
    )

    print(f"통합 데이터 크기: {integrated_df.shape}")

    # 2019-2023년 전체 기간 고용률 데이터가 있는 지자체 찾기
    complete_municipalities = []

    for (sido, sigungu), group in integrated_df.groupby(['시도', '시군구']):
        # 2019-2023년 모든 반기 데이터가 있고, 고용률이 NaN이 아닌 경우
        years_with_data = set()
        for _, row in group.iterrows():
            if pd.notna(row['고용률']) and row['고용률'] != 0:
                years_with_data.add(row['연도'])

        # 2019-2023년 모든 연도에 데이터가 있는 경우
        if years_with_data == {2019, 2020, 2021, 2022, 2023}:
            complete_municipalities.append((sido, sigungu))

    print(f"\\n2019-2023년 전체 기간 고용률 데이터가 있는 지자체: {len(complete_municipalities)}개")

    if len(complete_municipalities) > 0:
        print("완전한 데이터를 가진 지자체 (처음 10개):")
        for i, (sido, sigungu) in enumerate(complete_municipalities[:10], 1):
            print(f"{i:2d}. {sido} {sigungu}")

        # 완전한 데이터를 가진 지자체만 필터링
        mask = integrated_df.apply(
            lambda row: (row['시도'], row['시군구']) in complete_municipalities,
            axis=1
        )
        final_df = integrated_df[mask].copy()
    else:
        print("2019-2023년 전체 기간 데이터를 가진 지자체가 없습니다.")
        print("2021-2023년 데이터만 있는 지자체로 분석을 진행합니다.")

        # 2021-2023년 데이터가 있는 지자체 찾기
        complete_municipalities_2021 = []
        for (sido, sigungu), group in integrated_df.groupby(['시도', '시군구']):
            years_with_data = set()
            for _, row in group.iterrows():
                if pd.notna(row['고용률']) and row['고용률'] != 0:
                    years_with_data.add(row['연도'])

            if {2021, 2022, 2023}.issubset(years_with_data):
                complete_municipalities_2021.append((sido, sigungu))

        print(f"2021-2023년 데이터를 가진 지자체: {len(complete_municipalities_2021)}개")

        mask = integrated_df.apply(
            lambda row: (row['시도'], row['시군구']) in complete_municipalities_2021,
            axis=1
        )
        final_df = integrated_df[mask].copy()

        # 2021-2023년 데이터만 유지
        final_df = final_df[final_df['연도'].isin([2021, 2022, 2023])].copy()

    print(f"\\n최종 분석 데이터 크기: {final_df.shape}")

    # 결측값 처리
    final_df['E9_체류자수'] = final_df['E9_체류자수'].fillna(0)
    final_df['제조업_종사자수'] = final_df['제조업_종사자수'].fillna(0)
    final_df['전체_종사자수'] = final_df['전체_종사자수'].fillna(0)

    # 서비스업 종사자 수 계산 (전체 - 제조업)
    final_df['서비스업_종사자수'] = final_df['전체_종사자수'] - final_df['제조업_종사자수']
    final_df['서비스업_종사자수'] = final_df['서비스업_종사자수'].clip(lower=0)

    # 컬럼 순서 정리
    final_df = final_df[['시도', '시군구', '면적', '연도', '반기', 'E9_체류자수',
                         '제조업_종사자수', '서비스업_종사자수', '전체_종사자수', '고용률']].copy()

    return final_df

def main():
    """메인 실행 함수"""
    print("데이터 통합 시작...\\n")

    # 데이터 통합
    final_df = integrate_all_data()

    # 저장
    output_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/integrated_analysis_data.csv"
    final_df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\\n최종 통합 데이터 저장 완료: {output_path}")

    # 기본 통계 출력
    print("\\n=== 기본 통계 ===")
    print(f"지자체 수: {final_df[['시도', '시군구']].drop_duplicates().shape[0]}개")
    print(f"연도 범위: {final_df['연도'].min()}-{final_df['연도'].max()}")
    print(f"총 관측치: {len(final_df)}개")

    print("\\n연도별 평균 고용률:")
    print(final_df.groupby('연도')['고용률'].mean().round(2))

    print("\\n연도별 평균 E9 체류자 수:")
    print(final_df.groupby('연도')['E9_체류자수'].mean().round(0))

    return final_df

if __name__ == "__main__":
    result = main()