import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def load_employment_data():
    """고용률 데이터 로드 (2019-2023)"""
    print("=== 고용률 데이터 로드 ===")

    # Excel 파일 읽기
    file_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/raw/citycounty_gender_laborforce_summary.xlsx"
    df = pd.read_excel(file_path)

    print(f"원본 데이터 크기: {df.shape}")
    print("컬럼:", list(df.columns))
    print(df.head())

    # 데이터 정리
    # 첫 번째 행은 헤더 정보이므로 제거
    df = df.iloc[1:].copy()

    # 컬럼명 정리
    df.columns = ['행정구역', '성별'] + [col for col in df.columns[2:]]

    # '계'(전체) 데이터만 필터링
    df_total = df[df['성별'] == '계'].copy()

    print(f"\n'계' 데이터 크기: {df_total.shape}")
    print(df_total.head())

    return df_total

def load_e9_data():
    """E9 체류자 데이터 로드"""
    print("\n=== E9 체류자 데이터 로드 ===")

    file_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/raw/시군구별_E9_체류자(2014~2023).csv"

    # 인코딩 시도
    for encoding in ['cp949', 'euc-kr', 'utf-8']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"성공적으로 로드됨 (인코딩: {encoding})")
            break
        except:
            continue
    else:
        raise Exception("모든 인코딩 시도 실패")

    print(f"원본 데이터 크기: {df.shape}")
    print("컬럼:", list(df.columns))
    print(df.head())

    # 데이터 정리
    # 첫 번째 행은 헤더 정보이므로 제거
    df = df.iloc[1:].copy()

    # 전체 계(남+여) 데이터만 필터링 (성별이 '계'인 행)
    df_total = df[df.iloc[:, 2] == '계'].copy()

    print(f"\n'계' 데이터 크기: {df_total.shape}")
    print(df_total.head())

    return df_total

def load_area_data():
    """지역 면적 데이터 로드"""
    print("\n=== 지역 면적 데이터 로드 ===")

    file_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/raw/지역_면적_utf8.csv"
    df = pd.read_csv(file_path, encoding='utf-8')

    print(f"데이터 크기: {df.shape}")
    print("컬럼:", list(df.columns))
    print(df.head())

    return df

def load_industry_data():
    """산업별 종사자 데이터 로드"""
    print("\n=== 산업별 종사자 데이터 로드 ===")

    file_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/raw/지역별_산업별_종사자.csv"

    # 인코딩 시도
    for encoding in ['cp949', 'euc-kr', 'utf-8']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"성공적으로 로드됨 (인코딩: {encoding})")
            break
        except:
            continue
    else:
        raise Exception("모든 인코딩 시도 실패")

    print(f"데이터 크기: {df.shape}")
    print("컬럼:", list(df.columns))
    print(df.head(10))

    return df

def clean_region_names(df, region_col):
    """지역명 정리 (시도와 시군구 분리)"""
    print(f"\n=== 지역명 정리: {region_col} ===")

    # 지역명을 시도와 시군구로 분리
    regions = []
    for region in df[region_col]:
        if pd.isna(region):
            regions.append(('', ''))
            continue

        region = str(region).strip()

        # 시도별 패턴 매칭
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
            '제주': '제주특별자치도'
        }

        found_sido = None
        sigungu = region

        for pattern, full_sido in sido_patterns.items():
            if region.startswith(pattern):
                found_sido = full_sido
                sigungu = region[len(pattern):].strip()
                break

        # 시도가 없는 경우 (단독 시군구명)
        if found_sido is None:
            # 주변 시군구를 참고하여 시도 추정
            if '구' in region and region.endswith('구'):
                # 특별시/광역시의 구일 가능성
                found_sido = '미정'  # 나중에 처리
            else:
                found_sido = '미정'
            sigungu = region

        regions.append((found_sido, sigungu))

    # 결과를 데이터프레임에 추가
    df_copy = df.copy()
    df_copy['시도'] = [r[0] for r in regions]
    df_copy['시군구'] = [r[1] for r in regions]

    print(f"지역명 정리 완료: {len(df_copy)}개 지역")
    print("시도별 분포:")
    print(df_copy['시도'].value_counts())

    return df_copy

def main():
    """메인 전처리 함수"""
    print("데이터 전처리 시작...\n")

    # 각 데이터 로드
    employment_df = load_employment_data()
    e9_df = load_e9_data()
    area_df = load_area_data()
    industry_df = load_industry_data()

    # 저장
    output_dir = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed"
    os.makedirs(output_dir, exist_ok=True)

    employment_df.to_csv(f"{output_dir}/employment_raw.csv", index=False, encoding='utf-8')
    e9_df.to_csv(f"{output_dir}/e9_raw.csv", index=False, encoding='utf-8')
    area_df.to_csv(f"{output_dir}/area_raw.csv", index=False, encoding='utf-8')
    industry_df.to_csv(f"{output_dir}/industry_raw.csv", index=False, encoding='utf-8')

    print(f"\n원본 데이터 전처리 완료. 저장 위치: {output_dir}")

if __name__ == "__main__":
    main()