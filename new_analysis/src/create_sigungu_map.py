import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import os

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

        font_list = [f.name for f in fm.fontManager.ttflist if 'gothic' in f.name.lower() or 'malgun' in f.name.lower()]
        if font_list:
            plt.rcParams['font.family'] = font_list[0]
            plt.rcParams['axes.unicode_minus'] = False
            return True

        return False
    except Exception as e:
        print(f"폰트 설정 오류: {e}")
        return False

def load_and_convert_to_sigungu():
    """읍면동 단위 지도를 시군구 단위로 집계"""
    print("=== 시군구 단위 지도 생성 ===")

    # 기존 읍면동 지도 데이터 로드
    map_file = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/maps/korea_map.geojson"
    if not os.path.exists(map_file):
        print("❌ 지도 파일이 없습니다. 먼저 new_map_visualization.py를 실행하세요.")
        return None

    try:
        print("읍면동 지도 데이터 로드 중...")
        gdf = gpd.read_file(map_file)
        print(f"로드된 읍면동 수: {len(gdf)}")

        # 시군구별로 지오메트리 병합
        print("시군구 단위로 경계 병합 중...")

        # 시도와 시군구명을 결합하여 유니크한 시군구 식별자 생성
        gdf['시도_시군구'] = gdf['sidonm'] + '_' + gdf['sggnm']

        # 시군구별로 그룹화하여 지오메트리 병합
        from shapely.ops import unary_union

        def merge_geometries(geometries):
            return unary_union(geometries.tolist())

        sigungu_gdf = gdf.groupby(['sidonm', 'sggnm']).agg({
            'geometry': merge_geometries,   # 지오메트리 병합
            'sido': 'first',               # 시도 코드
            'sgg': 'first'                 # 시군구 코드
        }).reset_index()

        # GeoDataFrame으로 변환
        sigungu_gdf = gpd.GeoDataFrame(sigungu_gdf, geometry='geometry', crs=gdf.crs)

        print(f"생성된 시군구 수: {len(sigungu_gdf)}")

        # 컬럼명 정리
        sigungu_gdf = sigungu_gdf.rename(columns={
            'sidonm': '시도',
            'sggnm': '시군구'
        })

        # 시군구 저장
        output_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/maps/korea_sigungu.geojson"
        sigungu_gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')
        print(f"✅ 시군구 지도 저장: {output_path}")

        return sigungu_gdf

    except Exception as e:
        print(f"❌ 시군구 지도 생성 실패: {e}")
        return None

def test_sigungu_map_with_data():
    """우리 분석 데이터와 시군구 지도 연결 테스트"""
    print("\n=== 분석 데이터와 지도 연결 테스트 ===")

    # 1. 시군구 지도 로드
    sigungu_gdf = load_and_convert_to_sigungu()
    if sigungu_gdf is None:
        return

    # 2. 분석 데이터 로드
    data_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/new_processed/comprehensive_summary.csv"
    if not os.path.exists(data_path):
        print("❌ 분석 데이터가 없습니다.")
        return

    try:
        df = pd.read_csv(data_path)
        print(f"분석 데이터 로드: {len(df)}개 지자체")

        # 3. 지도와 데이터 매칭 확인
        map_regions = set(f"{row['시도']}_{row['시군구']}" for _, row in sigungu_gdf.iterrows())
        data_regions = set(f"{row['시도']}_{row['시군구']}" for _, row in df.iterrows())

        # 4. 매칭 결과
        matched = map_regions & data_regions
        map_only = map_regions - data_regions
        data_only = data_regions - map_regions

        print(f"\n📊 매칭 결과:")
        print(f"  지도 시군구 수: {len(map_regions)}")
        print(f"  데이터 시군구 수: {len(data_regions)}")
        print(f"  매칭된 시군구: {len(matched)}")
        print(f"  지도만 있음: {len(map_only)}")
        print(f"  데이터만 있음: {len(data_only)}")

        if len(data_only) > 0:
            print(f"\n⚠️ 지도에 없는 데이터 지역 (처음 10개):")
            for i, region in enumerate(sorted(list(data_only))[:10]):
                print(f"  {i+1}. {region}")

        # 5. 매칭 비율 계산
        match_ratio = len(matched) / len(data_regions) * 100
        print(f"\n✅ 매칭률: {match_ratio:.1f}%")

        if match_ratio >= 80:
            print("🎉 매칭률이 양호합니다!")
        else:
            print("⚠️ 매칭률이 낮습니다. 지역명 정리가 필요할 수 있습니다.")

        # 6. 테스트 시각화 - E9 체류자수로 색칠
        create_test_choropleth(sigungu_gdf, df, matched)

        return sigungu_gdf, df

    except Exception as e:
        print(f"❌ 데이터 연결 테스트 실패: {e}")
        return None, None

def create_test_choropleth(sigungu_gdf, df, matched_regions):
    """테스트용 단계구분도 생성 - E9 체류자수"""
    print("\n=== 테스트 단계구분도 생성 ===")

    try:
        setup_korean_font()

        # 1. 지도와 데이터 병합
        # 병합용 키 생성
        sigungu_gdf['merge_key'] = sigungu_gdf['시도'] + '_' + sigungu_gdf['시군구']
        df['merge_key'] = df['시도'] + '_' + df['시군구']

        # 매칭되는 데이터만 필터링
        matched_keys = [key for key in df['merge_key'] if key in sigungu_gdf['merge_key'].values]

        # 병합
        merged_gdf = sigungu_gdf.merge(df[['merge_key', 'E9_체류자수', '고용률']], on='merge_key', how='left')

        # 2. E9 체류자수 단계구분도
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # E9 체류자수 지도
        merged_gdf.plot(
            column='E9_체류자수',
            ax=ax1,
            cmap='Reds',
            linewidth=0.3,
            edgecolor='black',
            legend=True,
            legend_kwds={'shrink': 0.6, 'aspect': 20},
            missing_kwds={'color': 'lightgray'}
        )
        ax1.set_title('E9 체류자 수 (명)', fontsize=14, pad=20)
        ax1.axis('off')

        # 고용률 지도
        merged_gdf.plot(
            column='고용률',
            ax=ax2,
            cmap='Blues',
            linewidth=0.3,
            edgecolor='black',
            legend=True,
            legend_kwds={'shrink': 0.6, 'aspect': 20},
            missing_kwds={'color': 'lightgray'}
        )
        ax2.set_title('고용률 (%)', fontsize=14, pad=20)
        ax2.axis('off')

        plt.tight_layout()

        # 저장
        output_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/output"
        os.makedirs(output_path, exist_ok=True)

        plt.savefig(f"{output_path}/test_choropleth_map.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ 테스트 지도 저장: {output_path}/test_choropleth_map.png")
        print(f"📊 매칭된 지역 수: {len(matched_keys)}/{len(df)}")

    except Exception as e:
        print(f"❌ 단계구분도 생성 실패: {e}")
        print(f"오류 상세: {str(e)}")

def main():
    """메인 함수"""
    print("🗺️  시군구 단위 지도 생성 및 데이터 연결 테스트")
    print("=" * 60)

    # 시군구 지도 생성 및 데이터 연결 테스트
    result = test_sigungu_map_with_data()
    if result is None:
        print("❌ 테스트 실패")
        return None, None

    sigungu_gdf, df = result
    print("\n🎉 시군구 지도 테스트 완료!")
    return sigungu_gdf, df

if __name__ == "__main__":
    sigungu_gdf, df = main()