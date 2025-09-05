import pandas as pd
from pytrends.request import TrendReq
import time

# Hàm kiểm tra kết nối và chạy thử pytrends
def test_pytrends():
    try:
        # Khởi tạo pytrends
        print("Đang kết nối tới Google Trends...")
        pytrend = TrendReq(hl='en-US', tz=0)  # Ngôn ngữ: tiếng Anh, múi giờ: UTC
        
        # Kiểm tra Interest by Region với từ khóa "Hello World"
        print("\nLấy dữ liệu Interest by Region cho 'Hello World'...")
        pytrend.build_payload(kw_list=['Hello World'])
        df_region = pytrend.interest_by_region(resolution='COUNTRY')
        if not df_region.empty:
            print("Dữ liệu Interest by Region (5 dòng đầu tiên):")
            print(df_region.head())
        else:
            print("Không có dữ liệu Interest by Region.")

        # Kiểm tra Daily Trending Searches (US)
        time.sleep(1)  # Tránh giới hạn tốc độ
        print("\nLấy dữ liệu Daily Trending Searches tại Mỹ...")
        df_trending = pytrend.trending_searches(pn='united_states')
        if not df_trending.empty:
            print("Dữ liệu Trending Searches (5 dòng đầu tiên):")
            print(df_trending.head())
        else:
            print("Không có dữ liệu Trending Searches.")

        # Kiểm tra Keyword Suggestions cho "Hello World"
        time.sleep(1)
        print("\nLấy Keyword Suggestions cho 'Hello World'...")
        keywords = pytrend.suggestions(keyword='Hello World')
        df_keywords = pd.DataFrame(keywords).drop(columns='mid', errors='ignore')
        if not df_keywords.empty:
            print("Dữ liệu Keyword Suggestions (5 dòng đầu tiên):")
            print(df_keywords.head())
        else:
            print("Không có dữ liệu Keyword Suggestions.")

        print("\nKết luận: pytrends hoạt động bình thường!")
        
    except Exception as e:
        print(f"Lỗi: {e}")
        print("Kiểm tra kết nối mạng, cài đặt pytrends, hoặc giới hạn API từ Google Trends.")

# Chạy hàm kiểm tra
if __name__ == "__main__":
    print("Bắt đầu kiểm tra pytrends...")
    test_pytrends()