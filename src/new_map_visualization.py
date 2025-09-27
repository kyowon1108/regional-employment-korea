import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
import os
import requests
import zipfile

def setup_korean_font():
    """한글 폰트 설정"""
    try:
        # macOS의 기본 한글 폰트들 시도
        font_candidates = [
            '/System/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/AppleMyungjo.ttf',
            '/System/Library/Fonts/Arial Unicode MS.ttf'
        ]

        for font_path in font_candidates:
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                print(f"한글 폰트 설정 완료: {font_prop.get_name()}")
                return True

        # 시스템 폰트에서 한글 폰트 찾기
        font_list = [f.name for f in fm.fontManager.ttflist if 'gothic' in f.name.lower() or 'malgun' in f.name.lower()]
        if font_list:
            plt.rcParams['font.family'] = font_list[0]
            print(f"한글 폰트 설정 완료: {font_list[0]}")
            return True

        print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        return False

    except Exception as e:
        print(f"폰트 설정 중 오류: {e}")
        return False

def download_korea_shp():
    """한국 시군구 경계 shp 파일 다운로드"""
    print("=== 한국 시군구 경계 데이터 다운로드 ===")

    # 데이터 디렉토리 생성
    map_dir = "/Users/kapr/Desktop/DataAnalyze/new_analysis/data/maps"
    os.makedirs(map_dir, exist_ok=True)

    # shp 파일이 이미 있는지 확인
    shp_file = os.path.join(map_dir, "sig.shp")
    if os.path.exists(shp_file):
        print("✅ 지도 파일이 이미 존재합니다.")
        return shp_file

    # 여러 다운로드 URL 시도
    urls = [
        "https://github.com/southkorea/southkorea-maps/raw/master/kostat/2018/shp/skorea-municipalities-2018-shp.zip",
        "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2018/shp/skorea_municipalities_geo.json",
        "https://github.com/vuski/admdongkor/raw/master/ver20221001/HangJeongDong_ver20221001.geojson"
    ]

    for i, url in enumerate(urls):
        try:
            print(f"다운로드 시도 {i+1}: {url.split('/')[-1]}")

            if url.endswith('.geojson') or 'json' in url:
                # GeoJSON 파일 다운로드
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                geojson_path = os.path.join(map_dir, "korea_map.geojson")
                with open(geojson_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)

                print(f"✅ GeoJSON 데이터 다운로드 완료: {geojson_path}")
                return geojson_path

            else:
                # ZIP 파일 다운로드
                response = requests.get(url, timeout=60)
                response.raise_for_status()

                zip_path = os.path.join(map_dir, "korea_map.zip")
                with open(zip_path, 'wb') as f:
                    f.write(response.content)

                print("압축 해제 중...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(map_dir)

                # zip 파일 삭제
                os.remove(zip_path)

                # 압축 해제된 파일 중 .shp 파일 찾기
                for file in os.listdir(map_dir):
                    if file.endswith('.shp'):
                        old_path = os.path.join(map_dir, file)
                        new_path = os.path.join(map_dir, "sig.shp")
                        os.rename(old_path, new_path)

                        # 관련 파일들도 이름 변경
                        base_name = file[:-4]  # .shp 제거
                        for ext in ['.shx', '.dbf', '.prj']:
                            old_file = os.path.join(map_dir, base_name + ext)
                            new_file = os.path.join(map_dir, "sig" + ext)
                            if os.path.exists(old_file):
                                os.rename(old_file, new_file)

                        print(f"✅ 지도 데이터 다운로드 완료: {new_path}")
                        return new_path

                print("⚠️ shp 파일을 찾을 수 없습니다.")

        except Exception as e:
            print(f"❌ 다운로드 실패 {i+1}: {e}")
            continue

    print("❌ 모든 다운로드 시도 실패")
    print("📝 대안 방법:")
    print("1. https://github.com/southkorea/southkorea-maps 에서 수동 다운로드")
    print("2. 통계청 SGIS (https://sgis.kostat.go.kr) 에서 행정구역경계 다운로드")
    return None

def load_korea_map():
    """한국 지도 데이터 로드 및 좌표계 변환"""
    print("\n=== 한국 지도 데이터 로드 ===")

    map_file = download_korea_shp()
    if not map_file:
        return None

    try:
        print("지도 데이터 로드 중...")

        # 파일 형식에 따라 다른 방식으로 로드
        if map_file.endswith('.geojson'):
            # GeoJSON 파일 로드
            gdf = gpd.read_file(map_file)
            print("GeoJSON 파일로 로드됨")
        else:
            # SHP 파일 로드 (cp949 인코딩 시도)
            try:
                gdf = gpd.read_file(map_file, encoding='cp949')
                print("SHP 파일 (cp949)로 로드됨")
            except:
                gdf = gpd.read_file(map_file, encoding='utf-8')
                print("SHP 파일 (utf-8)로 로드됨")

        print(f"로드된 지역 수: {len(gdf)}")
        print(f"컬럼: {list(gdf.columns)}")

        # 좌표계 확인 및 변환
        print(f"원본 좌표계: {gdf.crs}")

        # 좌표계 변환
        if gdf.crs is None:
            # 한국 데이터의 경우 보통 EPSG:5179 또는 EPSG:4326
            print("좌표계가 설정되지 않음. EPSG:4326으로 설정")
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
            print(f"EPSG:4326으로 변환됨")

        print(f"최종 좌표계: {gdf.crs}")

        # 지역명 컬럼 확인
        print("\n컬럼별 샘플 데이터:")
        for col in gdf.columns:
            if col != 'geometry':
                sample_val = gdf[col].iloc[0] if len(gdf) > 0 else 'N/A'
                print(f"  {col}: {sample_val}")

        return gdf

    except Exception as e:
        print(f"❌ 지도 로드 실패: {e}")
        print(f"오류 상세: {str(e)}")
        return None

def test_basic_map_plot(gdf):
    """기본 지도 그리기 테스트"""
    print("\n=== 기본 지도 테스트 ===")

    setup_korean_font()

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # 전국 지도 그리기
    gdf.plot(ax=ax, linewidth=0.5, color='lightblue', edgecolor='black', alpha=0.7)

    plt.title('한국 시군구 경계 지도', fontsize=16, pad=20)
    plt.axis('off')

    # 지도 범위 설정 (한국 영토에 맞게)
    bounds = gdf.total_bounds
    ax.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
    ax.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)

    plt.tight_layout()

    # 저장
    output_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/output"
    os.makedirs(output_path, exist_ok=True)

    plt.savefig(f"{output_path}/korea_map_test.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ 지도 저장 완료: {output_path}/korea_map_test.png")

def test_regional_map(gdf):
    """특정 지역 지도 테스트 (경기도)"""
    print("\n=== 경기도 지역 지도 테스트 ===")

    # 경기도 지역 필터링 (시도명이나 코드로 필터링)
    # 컬럼명을 확인하여 적절한 필터링 조건 찾기
    region_cols = [col for col in gdf.columns if any(keyword in col.upper() for keyword in ['SIG', 'SIDO', 'NAME', 'NM'])]
    print(f"지역 관련 컬럼: {region_cols}")

    # 경기도 데이터 찾기
    gyeonggi_gdf = None
    for col in region_cols:
        if gdf[col].dtype == 'object':  # 문자열 컬럼만
            sample_values = gdf[col].head(10).values
            print(f"{col} 샘플: {sample_values}")

            # 경기도 관련 데이터 찾기
            gyeonggi_mask = gdf[col].str.contains('경기', na=False)
            if gyeonggi_mask.any():
                gyeonggi_gdf = gdf[gyeonggi_mask]
                print(f"✅ {col} 컬럼에서 경기도 {len(gyeonggi_gdf)}개 지역 발견")
                break

    if gyeonggi_gdf is not None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        gyeonggi_gdf.plot(ax=ax, linewidth=0.8, color='lightgreen', edgecolor='black', alpha=0.7)

        plt.title('경기도 시군구 지도', fontsize=14, pad=15)
        plt.axis('off')

        # 경기도 영역에 맞게 범위 조정
        bounds = gyeonggi_gdf.total_bounds
        ax.set_xlim(bounds[0] - 0.05, bounds[2] + 0.05)
        ax.set_ylim(bounds[1] - 0.05, bounds[3] + 0.05)

        plt.tight_layout()

        output_path = "/Users/kapr/Desktop/DataAnalyze/new_analysis/output"
        plt.savefig(f"{output_path}/gyeonggi_map_test.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ 경기도 지도 저장 완료: {output_path}/gyeonggi_map_test.png")
    else:
        print("⚠️ 경기도 데이터를 찾을 수 없습니다.")

def main():
    """메인 테스트 함수"""
    print("🗺️  새로운 한국 지도 시각화 테스트")
    print("=" * 50)

    # 1. 지도 데이터 로드
    gdf = load_korea_map()
    if gdf is None:
        print("❌ 지도 데이터 로드 실패")
        return

    # 2. 기본 지도 테스트
    test_basic_map_plot(gdf)

    # 3. 지역별 지도 테스트
    test_regional_map(gdf)

    print("\n🎉 지도 테스트 완료!")
    return gdf

if __name__ == "__main__":
    gdf = main()