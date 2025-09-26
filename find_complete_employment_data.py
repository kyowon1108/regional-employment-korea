import pandas as pd

# 2019-2020년 데이터 로드
df_2019_2020 = pd.read_csv('/Users/kapr/Desktop/DataAnalyze/data/processed/고용률_2019_2020_전처리완료.csv', encoding='utf-8')

# 2021-2023년 데이터 로드
df_2021_2023 = pd.read_csv('/Users/kapr/Desktop/DataAnalyze/data/processed/취업자_고용률_전처리완료_utf-8.csv', encoding='utf-8')

print("2019-2020년 데이터 구조:")
print(df_2019_2020.head())
print(f"\n데이터 크기: {df_2019_2020.shape}")
print(f"연도별 데이터 수:")
print(df_2019_2020.groupby(['연도', '반기']).size())

print("\n" + "="*50)

print("2021-2023년 데이터 구조:")
print(df_2021_2023.head())
print(f"\n데이터 크기: {df_2021_2023.shape}")
print(f"연도별 데이터 수:")
print(df_2021_2023.groupby(['연도', '반기']).size())

# 2019-2020년 데이터에서 고용률이 있는 지자체 추출
municipalities_2019_2020 = set(df_2019_2020['시군구'].unique())
print(f"\n2019-2020년 고용률 데이터가 있는 지자체 수: {len(municipalities_2019_2020)}")

# 2021-2023년 데이터에서 고용률이 있는 지자체 추출 (시도+시군구 결합)
df_2021_2023['지자체명'] = df_2021_2023['시도'] + ' ' + df_2021_2023['시군구']
municipalities_2021_2023 = set(df_2021_2023['시군구'].unique())
print(f"2021-2023년 고용률 데이터가 있는 지자체 수: {len(municipalities_2021_2023)}")

# 두 기간 모두에 있는 지자체 찾기
common_municipalities = municipalities_2019_2020.intersection(municipalities_2021_2023)
print(f"\n2019-2023년 전체 기간 고용률 데이터가 있는 지자체 수: {len(common_municipalities)}")

# 각 지자체별로 전체 기간의 데이터 완전성 확인
complete_data_municipalities = []

for municipality in common_municipalities:
    # 2019-2020년 데이터 확인 (4개 반기: 2019년 1,2반기, 2020년 1,2반기)
    data_2019_2020 = df_2019_2020[df_2019_2020['시군구'] == municipality]
    expected_periods_2019_2020 = [(2019, 1), (2019, 2), (2020, 1), (2020, 2)]
    actual_periods_2019_2020 = set(zip(data_2019_2020['연도'], data_2019_2020['반기']))

    # 2021-2023년 데이터 확인 (6개 반기: 2021-2023년 각 1,2반기)
    data_2021_2023 = df_2021_2023[df_2021_2023['시군구'] == municipality]
    expected_periods_2021_2023 = [(2021, 1), (2021, 2), (2022, 1), (2022, 2), (2023, 1), (2023, 2)]
    actual_periods_2021_2023 = set(zip(data_2021_2023['연도'], data_2021_2023['반기']))

    # 모든 기간의 데이터가 완전한지 확인
    if (set(expected_periods_2019_2020).issubset(actual_periods_2019_2020) and
        set(expected_periods_2021_2023).issubset(actual_periods_2021_2023)):
        complete_data_municipalities.append(municipality)

print(f"\n2019-2023년 전체 기간 완전한 고용률 데이터를 가진 지자체 수: {len(complete_data_municipalities)}")

# 완전한 데이터를 가진 지자체 목록 출력
print("\n완전한 고용률 데이터를 가진 지자체 목록:")
complete_data_municipalities.sort()
for i, municipality in enumerate(complete_data_municipalities, 1):
    print(f"{i:3d}. {municipality}")

# 결과를 CSV 파일로 저장
result_df = pd.DataFrame({
    '순번': range(1, len(complete_data_municipalities) + 1),
    '지자체명': complete_data_municipalities
})
result_df.to_csv('/Users/kapr/Desktop/DataAnalyze/완전한_고용률_데이터_지자체_목록.csv',
                index=False, encoding='utf-8')
print(f"\n결과가 '완전한_고용률_데이터_지자체_목록.csv' 파일로 저장되었습니다.")