import pandas as pd
from data_loading import get_monthly_traffic, fetch_bigquery_table
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlparse

@st.cache_data
def load_data():
    df = get_monthly_traffic()
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(by='date', ascending=True)

@st.cache_data
def fetch_news_data():
    df = fetch_bigquery_table("pr-project-444202.pr_project_scraped_data.news_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    return df.sort_values(by='Date', ascending=True)

@st.cache_data
def fetch_web_data():
    df= fetch_bigquery_table("pr-project-444202.pr_project_scraped_data.web_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    df = df.dropna(subset=['Date'])
    return df.sort_values(by='Date', ascending=True)

@st.cache_data
def fetch_twitter_data():
    df = fetch_bigquery_table("pr-project-444202.pr_project_scraped_data.twitter_data")
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.date
    df = df.dropna(subset=['created_at'])
    return df.sort_values(by='created_at', ascending=True)

@st.cache_data
def fetch_facebook_data():
    df = fetch_bigquery_table("pr-project-444202.pr_project_scraped_data.facebook_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    df = df.dropna(subset=['Date'])
    return df.sort_values(by='Date', ascending=True)

@st.cache_data
def fetch_instagram_data():
    df = fetch_bigquery_table("pr-project-444202.pr_project_scraped_data.instagram_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    df = df.dropna(subset=['Date'])
    return df.sort_values(by='Date', ascending=True)
    
@st.cache_data
def fetch_youtube_data():
    df = fetch_bigquery_table("pr-project-444202.pr_project_scraped_data.youtube_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    df = df.dropna(subset=['Date'])
    df['AVE'] = df['Views']*0.2
    return df.sort_values(by='Date', ascending=True)

def filter_mentions(df, start_date, end_date, source_label, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    filtered_df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    
    mentions_per_day = df.groupby(date_col).size().reset_index(name=source_label)
    mentions_per_day.rename(columns={date_col: "Date"}, inplace=True)
    
    mentions_per_day["Date"] = pd.to_datetime(mentions_per_day["Date"]).dt.date

    return mentions_per_day

def filter_sentiment(df, date_col, sentiment_col):
    df[date_col] = pd.to_datetime(df[date_col])
    conditions = [
        (df[sentiment_col] > 0.5), 
        (df[sentiment_col] < 0),
        (df[sentiment_col] >= 0) & (df[sentiment_col] <= 0.5)
    ]
    choices = ['positive', 'negative', 'neutral']
    df['category'] = np.select(conditions, choices, default='neutral') 

    sentiment_counts = df.groupby([pd.Grouper(key=date_col, freq='D'), 'category']).size().unstack(fill_value=0)
    return sentiment_counts

traffic_df = load_data()
news_df = fetch_news_data()
web_df = fetch_web_data()
twt_df = fetch_twitter_data()
fb_df = fetch_facebook_data()
ig_df = fetch_instagram_data()
yt_df = fetch_youtube_data()

twt_df['total_engagements'] = twt_df['retweets'] + twt_df['likes'] + twt_df['views'] + twt_df['reply_count']
twt_df['AVE'] = twt_df['total_engagements']*0.15

fb_df['Reach'] = fb_df[['Likes', 'Comments', 'Shares']].max(axis=1)
fb_df['AVE'] = (fb_df['Likes'] *  0.05 ) + (fb_df['Comments'] * 0.10) + (fb_df['Shares'] * 0.10)

ig_df['Reach'] = ig_df[['Likes', 'Comments']].max(axis=1)
ig_df['AVE'] = (ig_df['Likes'] * 0.1) + (ig_df['Comments'] * 0.2)

def count_mentions(df, start_date, end_date, date_col):
    filtered_df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    return len(filtered_df)

def count_reach(df, start_date, end_date, date_col, reach_col):
    filtered_df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    filtered_df[reach_col] = filtered_df[reach_col].fillna(0)
    return int(sum(filtered_df[reach_col].values))

def count_ave(df, start_date, end_date, date_col):
    filtered_df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    filtered_df['AVE'] = filtered_df['AVE'].fillna(0)
    return int(sum(filtered_df['AVE'].values))

st.markdown("""
<style>
    .big-metric {
        text-align: center;
        padding: 20px;
        border: 2px solid gray;
        border-radius: 10px;
        margin: 10px;
        width: 90%;  /* Control the width of the metric box */
        margin-left: auto;
        margin-right: auto;
    }
    .big-metric-sent {
        text-align: center;
        padding: 14px;
        border: 2px solid gray;
        border-radius: 10px;
        margin: 10px;
        width: 90%;  /* Control the width of the metric box */
        margin-left: auto;
        margin-right: auto;
    }
    .metric-title {
        color: black;
        margin-bottom: 5px;  /* Space between the title and the numbers */
    }
    .metric-number {
        color: black;
        font-size: 28px;  /* Size of the number in engagement metrics */
    }
    .sentiment-metrics {
        color: #21ba45;  /* Green for positive */
        font-size: 18px;  /* Smaller font size for sentiment metrics */
    }
    .sentiment-negative {
        color: #db2828;  /* Red for negative */
        font-size: 18px;  /* Smaller font size for sentiment metrics */
    }
</style>
""", unsafe_allow_html=True)

traffic_df['date'] = pd.to_datetime(traffic_df['date'])
traffic_df = traffic_df.sort_values(by='date', ascending=True)

# Get the min and max date from the dataframe
min_date = traffic_df['date'].min()
max_date = traffic_df['date'].max()

# Time range selection
st.title("Shinta Mani PR Monitoring Dashboard")
time_range = st.selectbox(
    "Select Time Range",
    options=["Past Week", "Past Month", "Past Year","Custom"],
    index=1
)

# Calculate date range based on selection
if time_range == "Past Week":
    start_date, end_date = max_date - timedelta(days=7), max_date
    hoverdistance=25
elif time_range == "Past Month":
    start_date, end_date = max_date - timedelta(days=30), max_date
    hoverdistance=10
elif time_range == "Past Year":
    start_date, end_date = max_date - timedelta(days=365), max_date
    hoverdistance=2
elif time_range == "Custom":
    # Use a select slider for a custom range
    start_date, end_date = st.select_slider(
        "Select Custom Date Range",
        options=list(traffic_df['date']),
        value=(min_date, max_date),
        format_func=lambda x: x.strftime('%Y-%m-%d')
    )
    hoverdistance=1

# Filter data based on selected date range
filtered_data = traffic_df[(traffic_df['date'] >= start_date) & (traffic_df['date'] <= end_date)]
news_df['Date'] = pd.to_datetime(news_df['Date'])
news_df = news_df[(news_df['Date'] >= start_date) & (news_df['Date'] <= end_date)]

web_df['Date'] = pd.to_datetime(web_df['Date'])
web_df = web_df[(web_df['Date'] >= start_date) & (web_df['Date'] <= end_date)]

twt_df['created_at'] = pd.to_datetime(twt_df['created_at'])
twt_df = twt_df[(twt_df['created_at'] >= start_date) & (twt_df['created_at'] <= end_date)]

fb_df['Date'] = pd.to_datetime(fb_df['Date'])
fb_df = fb_df[(fb_df['Date'] >= start_date) & (fb_df['Date'] <= end_date)]

ig_df['Date'] = pd.to_datetime(ig_df['Date'])
ig_df = ig_df[(ig_df['Date'] >= start_date) & (ig_df['Date'] <= end_date)]

yt_df['Date'] = pd.to_datetime(yt_df['Date'])
yt_df = yt_df[(yt_df['Date'] >= start_date) & (yt_df['Date'] <= end_date)]

total_mentions = {
    "News" : count_mentions(news_df, start_date, end_date, 'Date'),
    "Web & Blogs" : count_mentions(web_df, start_date, end_date, 'Date'),
    "Twitter" : count_mentions(twt_df, start_date, end_date, 'created_at'),
    "Facebook" : count_mentions(fb_df, start_date, end_date, 'Date'),
    "Instagram" : count_mentions(ig_df, start_date, end_date, 'Date'),
    "YouTube" : count_mentions(yt_df, start_date, end_date, 'Date'),
    }

total_reach = {
    "News" : count_reach(news_df, start_date, end_date, 'Date', 'Estimated Reach'),
    "Web & Blogs" : count_reach(web_df, start_date, end_date, 'Date', 'Estimated Reach'),
    "Twitter" : count_reach(twt_df, start_date, end_date, 'created_at','views'),
    "Facebook": count_reach(fb_df, start_date, end_date, 'Date', 'Reach'),
    'Instagram': count_reach(ig_df, start_date, end_date, 'Date', 'Reach'),
    "YouTube" : count_reach(yt_df, start_date, end_date, 'Date','Views')
    }

total_ave = {
    "News" : count_ave(news_df, start_date, end_date, 'Date'),
    "Web & Blogs" : count_ave(web_df, start_date, end_date, 'Date'),
    "Twitter" : count_ave(twt_df, start_date, end_date, 'created_at'),
    "Facebook" : count_ave(fb_df, start_date, end_date, 'Date'),
    "Instagram" : count_ave(ig_df, start_date, end_date, 'Date'),
    "YouTube" : count_ave(yt_df, start_date, end_date, 'Date')
    }



col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="big-metric">
        <div class="metric-title">Total Mentions:</div>
        <div class="metric-number">{sum(total_mentions.values())}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="big-metric">
        <div class="metric-title">Total Reach:</div>
        <div class="metric-number">{sum(total_reach.values())}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="big-metric">
        <div class="metric-title">Total AVE</div>
        <div class="metric-number">{sum(total_ave.values())} USD</div>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("See Breakdown by Platform"):
        for platform, count in total_mentions.items():
            st.write(f"{platform}: {count} mentions")

with col2:
    with st.expander("See Breakdown by Platform"):
        for platform, count in total_reach.items():
            st.write(f"{platform}: {count}")

with col3:
    with st.expander("See Breakdown by Platform"):
        for platform, count in total_ave.items():
            st.write(f"{platform}: {count} USD")


news_mentions = filter_mentions(news_df, start_date, end_date, "News", "Date")
web_mentions = filter_mentions(web_df, start_date, end_date, "Web & Blogs", "Date")
twt_mentions = filter_mentions(twt_df, start_date, end_date, "Twitter", "created_at")
fb_mentions = filter_mentions(fb_df, start_date, end_date, "Facebook", "Date")
ig_mentions = filter_mentions(ig_df, start_date, end_date, "Instagram", "Date")
yt_mentions = filter_mentions(yt_df, start_date, end_date, "Youtube", "Date")

full_date_range = pd.DataFrame({
        "Date": pd.date_range(start = start_date, end = end_date, freq="D")
    })
full_date_range["Date"] = pd.to_datetime(full_date_range["Date"]).dt.date

mentions_combined_df = full_date_range.merge(news_mentions, on="Date", how="left")
mentions_combined_df = mentions_combined_df.merge(web_mentions, on="Date", how="left")
mentions_combined_df = mentions_combined_df.merge(twt_mentions, on="Date", how="left")
mentions_combined_df = mentions_combined_df.merge(fb_mentions, on="Date", how="left")
mentions_combined_df = mentions_combined_df.merge(ig_mentions, on="Date", how="left")
mentions_combined_df = mentions_combined_df.merge(yt_mentions, on="Date", how="left")

mentions_combined_df.fillna(0, inplace=True)
mentions_combined_df["Twitter"] = mentions_combined_df["Twitter"].astype(int)
mentions_combined_df["News"] = mentions_combined_df["News"].astype(int)
mentions_combined_df["Web & Blogs"] = mentions_combined_df["Web & Blogs"].astype(int)
mentions_combined_df["Facebook"] = mentions_combined_df["Facebook"].astype(int)
mentions_combined_df["Instagram"] = mentions_combined_df["Instagram"].astype(int)
mentions_combined_df["Youtube"] = mentions_combined_df["Youtube"].astype(int)
mentions_combined_df['Date'] = pd.to_datetime(mentions_combined_df['Date'])
plot_df = mentions_combined_df    


col1, col2 = st.columns([3,1])


with col2:
    granularity = st.selectbox("Select Granularity:", ["Daily", "Weekly", "Monthly"],key="granularity1")

    if granularity == "Daily":
        plot_df = mentions_combined_df
    elif granularity == "Weekly":
        plot_df = mentions_combined_df.resample("W-Mon", on="Date").sum().reset_index()
    elif granularity == "Monthly":
        plot_df = mentions_combined_df.resample("M", on="Date").sum().reset_index()

    sources = plot_df.columns[1:]  # Assuming the first column is 'Date'
    selected_sources = []
    for source in sources:
        if st.checkbox(source, True, key = str(source) + "1"):  # Default all selected
            selected_sources.append(source)



# Traces for each channel
channels = ['News', 'Web & Blogs','Twitter',  'Facebook', 'Instagram', 'Youtube']

colors = {
    'Twitter': (0,0,0), #, (29, 161, 242),  # Twitter blue
    'News': (0, 0, 255),        # Standard blue for generic news
    'Web & Blogs': (0, 128, 0), # Darker green for a professional look
    'Facebook': (66, 103, 178), # Facebook blue
    'Instagram': (193, 53, 132),# Instagram gradient primary color (magenta)
    'Youtube': (255, 0, 0),     # YouTube red
    'positive': (0, 128, 0),  # Green
    'negative': (255, 0, 0)   # Red
}

fig = go.Figure()

for channel in channels:

    color_rgb = colors[channel]
    fillcolor = f'rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},0.2)'

    if channel not in selected_sources:
        continue
    fig.add_trace(go.Scatter(
        x=plot_df["Date"],
        y=plot_df[channel],
        mode='lines+markers',
        name=channel,
        line=dict(shape='spline', color=f'rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})', width=2),
        marker=dict(size=6, color=f'rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})', symbol='circle'),
        fill='tozeroy',
        fillcolor=fillcolor,
        hovertemplate=f'<b>%{{fullData.name}}</b><br>Date: %{{x}}<br>Value: %{{y}}<extra></extra>'
    ))

# Configure the axes
max_value = plot_df.select_dtypes(include=[np.number]).max().max()
axis_mode = st.sidebar.radio("Axis Mode:", ["Fixed", "Dynamic"])

if axis_mode == "Fixed":
    fig.update_layout(
        xaxis=dict(showgrid=False, fixedrange=True),
        yaxis=dict(showgrid=True, fixedrange=True, range=[0, max_value + 5])
    )
else:
    fig.update_layout(
        xaxis=dict(showgrid=False, automargin=True, fixedrange=False),
        yaxis=dict(showgrid=True, automargin=True, fixedrange=False, rangemode="tozero")
    )

# Customize layout
fig.update_layout(
    title = "Mentions by Platform",
    xaxis_title="Date",
    yaxis_title="Mentions",
    template="plotly_white",
    hovermode='x unified',
    showlegend=False,
    height = 400
)

with col1:
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Social Listening Metrics")
col1, col2 = st.columns(2)


twt_df['Total_Engagement'] = twt_df[['retweets', 'likes', 'reply_count']].sum(axis=1)
fb_df['Total_Engagement'] = fb_df[['Likes', 'Comments', 'Shares']].sum(axis=1)
ig_df['Total_Engagement'] = ig_df[['Likes', 'Comments']].sum(axis=1)
yt_df['Total_Engagement'] = yt_df[['Likes', 'Number of Comments']].sum(axis=1)

total_engagement_twitter = twt_df[['retweets', 'likes', 'reply_count']].sum().sum()
total_engagement_facebook = fb_df[['Likes', 'Comments', 'Shares']].sum().sum()
total_engagement_instagram = ig_df[['Likes', 'Comments']].sum().sum()
total_engagement_youtube = yt_df[['Likes', 'Number of Comments']].sum().sum()

likes = (twt_df['likes'].sum() + fb_df['Likes'].sum() + 
         ig_df['Likes'].sum() + yt_df['Likes'].sum())

shares = (twt_df['retweets'].sum() + fb_df['Shares'].sum())

comments = (twt_df['reply_count'].sum() + fb_df['Comments'].sum() +
            ig_df['Comments'].sum() + yt_df['Number of Comments'].sum())

total_engagement_by_type = {
    "Likes": likes,
    "Shares": shares,
    "Comments": comments
}

engagement_df = pd.DataFrame({
        "Date": pd.date_range(start = start_date, end = end_date, freq="D")
    })
engagement_df["Date"] = pd.to_datetime(engagement_df["Date"]).dt.date

twt_df['created_at'] = pd.to_datetime(twt_df['created_at']).dt.date
fb_df['Date'] = pd.to_datetime(fb_df['Date']).dt.date
ig_df['Date'] = pd.to_datetime(ig_df['Date']).dt.date
yt_df['Date'] = pd.to_datetime(yt_df['Date']).dt.date

twt_agg = twt_df.groupby('created_at')['Total_Engagement'].sum().reset_index().rename(columns={'created_at': 'Date'})
fb_agg = fb_df.groupby('Date')['Total_Engagement'].sum().reset_index()
ig_agg = ig_df.groupby('Date')['Total_Engagement'].sum().reset_index()
yt_agg = yt_df.groupby('Date')['Total_Engagement'].sum().reset_index()

engagement_df = engagement_df.merge(twt_agg, on='Date', how='left', suffixes=('', '_twt'))
engagement_df = engagement_df.merge(fb_agg, on='Date', how='left', suffixes=('', '_fb'))
engagement_df = engagement_df.merge(ig_agg, on='Date', how='left', suffixes=('', '_ig'))
engagement_df = engagement_df.merge(yt_agg, on='Date', how='left', suffixes=('', '_yt'))

engagement_df.fillna(0, inplace=True)
engagement_df.columns = ['Date', 'Twitter', 'Facebook', 'Instagram', 'Youtube']
engagement_df['Date'] = pd.to_datetime(engagement_df['Date'])  # Ensure 'Date' is of datetime type

total_engagement = likes + shares + comments

with col1:
    st.markdown(f"""
    <div class="big-metric">
        <div class="metric-title">Social Engagement:</div>
        <div class="metric-number">{total_engagement}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("See Breakdown by Interaction Type"):
        for interaction_type, count in total_engagement_by_type.items():
            st.write(f"{interaction_type}: {count} interactions")

    with st.expander("See Breakdown by Platform"):
        st.write(f"Twitter: {total_engagement_twitter} Engagements")
        st.write(f"Facebook: {total_engagement_facebook} Engagements")
        st.write(f"Instagram: {total_engagement_instagram} Engagements")
        st.write(f"Youtube: {total_engagement_youtube} Engagements")



dates = pd.date_range(start=start_date, end=end_date)
data = {'Date': dates, 'positive': 0, 'negative': 0, 'neutral': 0}
sentiment_data = pd.DataFrame(data).set_index('Date')

news_sentiment = filter_sentiment(news_df, "Date", "Sentiment Score")
web_sentiment = filter_sentiment(web_df, "Date", "Sentiment Score")
twt_sentiment = filter_sentiment(twt_df, "created_at", "sentiment")
fb_sentiment = filter_sentiment(fb_df, "Date", "Sentiment")
ig_sentiment = filter_sentiment(ig_df, "Date", "Sentiment")
yt_sentiment = filter_sentiment(yt_df, "Date", "Sentiment")

def update_sentiment(master_df, updates):
    for col in updates.columns:
        if col in master_df.columns:
            master_df[col] += updates[col].reindex(master_df.index, fill_value=0)
    return master_df

sentiment_data = update_sentiment(sentiment_data, news_sentiment)
sentiment_data = update_sentiment(sentiment_data, web_sentiment)
sentiment_data = update_sentiment(sentiment_data, twt_sentiment)
sentiment_data = update_sentiment(sentiment_data, fb_sentiment)
sentiment_data = update_sentiment(sentiment_data, ig_sentiment)
sentiment_data = update_sentiment(sentiment_data, yt_sentiment)

total_positive = sentiment_data['positive'].sum()
total_negative = sentiment_data['negative'].sum()
total_neutral = sentiment_data['neutral'].sum()

total_sentiments = total_positive + total_negative + total_neutral

positive_percentage = (total_positive / total_sentiments) * 100
negative_percentage = (total_negative / total_sentiments) * 100
neutral_percentage = (total_neutral / total_sentiments) * 100

with col2:
    st.markdown(f"""
    <div class="big-metric-sent">
        <div class="metric-title">Sentiment Analysis</div>
        <div class="sentiment-metrics">Positive: {total_positive}</div>
        <div class="sentiment-negative">Negative: {total_negative}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("See Sentiment Breakdown"):
        st.write(f"Total Positive: {total_positive} ({positive_percentage:.2f}%)")
        st.write(f"Total Negative: {total_negative} ({negative_percentage:.2f}%)")
        st.write(f"Total Neutral: {total_neutral} ({neutral_percentage:.2f}%)")

col1, col2 = st.columns([3,1])
with col2:
    granularity = st.selectbox("Select Granularity:", ["Daily", "Weekly", "Monthly"],key="granularity2")

    if granularity == "Daily":
        engagement_df = engagement_df
    elif granularity == "Weekly":
        engagement_df = engagement_df.resample("W-Mon", on="Date").sum().reset_index()
    elif granularity == "Monthly":
        engagement_df = engagement_df.resample("M", on="Date").sum().reset_index()

    sources = ['Twitter', 'Facebook','Instagram', 'Youtube']  # Assuming the first column is 'Date'
    selected_sources = []
    for source in sources:
        if st.checkbox(source, True):  # Default all selected
            selected_sources.append(source)

fig = go.Figure()

for channel in ['Twitter', 'Facebook', 'Instagram', 'Youtube']:
    color_rgb = colors[channel]
    fillcolor = f'rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},0.2)'

    if channel not in selected_sources:
        continue
    fig.add_trace(go.Scatter(
        x=engagement_df['Date'],
        y=engagement_df[channel],
        mode='lines+markers',
        name=channel,
        line=dict(shape='spline', color=f'rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})', width=2),
        marker=dict(size=6, color=f'rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})', symbol='circle'),
        fill='tozeroy',
        fillcolor=fillcolor,
        hovertemplate=f'<b>{channel}</b><br>Date: %{{x}}<br>Value: %{{y}}<extra></extra>'
    ))

fig.update_layout(
    title = "Social Engagement",
    xaxis_title="Date",
    yaxis_title="Total Engagement",
    template="plotly_white",
    hovermode='x unified',
    showlegend=False,
    height = 300
)

max_value = engagement_df.select_dtypes(include=[np.number]).max().max()

if axis_mode == "Fixed":
    fig.update_layout(
        xaxis=dict(showgrid=False, fixedrange=True),
        yaxis=dict(showgrid=True, fixedrange=True, range=[0, max_value + 5])
    )
else:
    fig.update_layout(
        xaxis=dict(showgrid=False, automargin=True, fixedrange=False),
        yaxis=dict(showgrid=True, automargin=True, fixedrange=False, rangemode="tozero")
    )
    
with col1:
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns([3,1])
with col2:
    granularity = st.selectbox("Select Granularity:", ["Daily", "Weekly", "Monthly"],key="granularity3")

    if granularity == "Daily":
        engagement_df = engagement_df
    elif granularity == "Weekly":
        engagement_df = engagement_df.resample("W-Mon", on="Date").sum().reset_index()
    elif granularity == "Monthly":
        engagement_df = engagement_df.resample("M", on="Date").sum().reset_index()

    sources = ['Positive', 'Negative']
    selected_sources = []
    for source in sources:
        if st.checkbox(source, True):  # Default all selected
            selected_sources.append(source.lower())

fig = go.Figure()
for channel in ['positive','negative']:
    color_rgb = colors[channel]
    fillcolor = f'rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},0.2)'

    if channel not in selected_sources:
        continue
    fig.add_trace(go.Scatter(
        x=sentiment_data.index,
        y=sentiment_data[channel],
        mode='lines+markers',
        name=channel,
        line=dict(shape='spline', color=f'rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})', width=2),
        marker=dict(size=6, color=f'rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})', symbol='circle'),
        fill='tozeroy',
        fillcolor=fillcolor,
        hovertemplate=f'<b>{channel}</b><br>Date: %{{x}}<br>Value: %{{y}}<extra></extra>'
    ))

fig.update_layout(
    title = "Sentiment Trend",
    xaxis_title="Date",
    yaxis_title="Sentiment",
    template="plotly_white",
    hovermode='x unified',
    showlegend=False,
    height = 300
)

with col1:
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Traffic Data")
col1, col2 = st.columns(2)
filtered_df = traffic_df[(traffic_df['date'] >= start_date) & (traffic_df['date'] <= end_date)]

average_per_day = filtered_df['total_page_views'].mean()
average_per_day = int(average_per_day)

total_views = filtered_df['total_page_views'].sum()

filtered_df['percentage_change'] = filtered_df['total_page_views'].pct_change() * 100
average_percentage_change = filtered_df['percentage_change'].mean()
average_percentage_change = round(average_percentage_change,2)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="big-metric">
        <div class="metric-title">Total Traffic:</div>
        <div class="metric-number">{total_views}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="big-metric">
        <div class="metric-title">Average Traffic per Day:</div>
        <div class="metric-number">{average_per_day}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="big-metric">
        <div class="metric-title">Average Traffic Change</div>
        <div class="metric-number">{average_percentage_change} %</div>
    </div>
    """, unsafe_allow_html=True)

graph_placeholder = st.empty()

options = ['News', 'Web & Blogs', 'Twitter', 'Facebook', 'Instagram', 'Youtube']

with st.container():
    cols = st.columns(len(options))
    selected_options = [cols[i].checkbox(options[i]) for i in range(len(options))]
    
fig = go.Figure()

# Add the main line for total page views
fig.add_trace(go.Scatter(
    x=filtered_data['date'],
    y=filtered_data['total_page_views'],
    mode='lines',
    name='Total Page Views',
    hovertemplate='Total Page Views: %{y}<extra></extra>',
    showlegend= True  # Ensure this is true if you want it in the legend
))

fig.update_layout(
    title="Website Traffic Over Time",  # Title of the graph
    xaxis_title='Date',  # Label for the X-axis
    yaxis_title='Total Page Views',  # Label for the Y-axis
    hovermode='x unified',
    showlegend = True,
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
        )
)

news_df['Display Text'] = news_df['Web'] + ' - ' + news_df['Title'].str[:10] + '...'
news_grouped = news_df.groupby(news_df['Date'].dt.date).agg({
    'Display Text': lambda x: ' | '.join(x)  # Aggregate display texts
}).reset_index()
news_grouped['Date'] = pd.to_datetime(news_grouped['Date'])

over_limit = False
if len(news_grouped) > 30:
    over_limit = True
    news_grouped = news_grouped.iloc[np.linspace(0, len(news_grouped) - 1, 30, dtype=int)]


if selected_options[0]:
    if over_limit:
        st.write("There are more than 30 mentions for News articles, some mentions will be omitted for better visualization. Select a smaller time frame to display all mentions.")

    color_rgb = colors["News"]
    fillcolor = f'rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},1)'
    
    for _, event in news_grouped.iterrows():
        #event_date = event['Date'].timestamp() * 1000
        fig.add_trace(go.Scatter(
            x=[event['Date']],
            y=[filtered_data['total_page_views'].max() * 1.1],  # slightly above the max value for visibility
            mode='markers',
            marker=dict(size=0, color='white'),
            #hoverinfo='text',
            hovertemplate="<b>PR Mention: " + event['Display Text'] + "(" + str(event['Date'].date()) + ")<extra></extra>",
            showlegend = False
        ))

        # Add a vertical line for visual aid
        #st.write("date being passed",event_date)
        fig.add_vline(x=event['Date'], line=dict(color=fillcolor, width=1, dash="dash"))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color=fillcolor, dash='dot', width=2),
        name='News'
    ))                

def create_display_text(row):
    domain = urlparse(row['Link']).netloc.replace("www.", "")
    if pd.isna(row['Title']) or row['Title'] == "":
        return domain  # Return just the domain if title is empty or NaN
    
    title_part = row['Title'][:10]
    # Add ellipsis only if the title is longer than 10 characters
    if len(row['Title']) > 10:
        title_part += '...'
    return f"{domain} - {title_part}"

if (len(web_df) > 0):
    web_df['Display Text'] = web_df.apply(create_display_text, axis=1)
    web_grouped = web_df.groupby(web_df['Date'].dt.date).agg({
        'Display Text': lambda x: ' | '.join(x)  # Aggregate display texts
    }).reset_index()
    web_grouped['Date'] = pd.to_datetime(web_grouped['Date'])
else:
    web_grouped = pd.DataFrame()

over_limit = False
if len(web_grouped) > 30:
    over_limit = True
    web_grouped = web_grouped.iloc[np.linspace(0, len(web_grouped) - 1, 30, dtype=int)]


if selected_options[1]:
    if over_limit:
        st.write("There are more than 30 mentions for Web & Blogs, some mentions will be omitted for better visualization. Select a smaller time frame to display all mentions.")

    color_rgb = colors["Web & Blogs"]
    fillcolor = f'rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},1)'
    for _, event in web_grouped.iterrows():
        #event_date = event['Date'].timestamp() * 1000
        fig.add_trace(go.Scatter(
            x=[event['Date']],
            y=[filtered_data['total_page_views'].max() * 1.1],  # slightly above the max value for visibility
            mode='markers',
            marker=dict(size=0, color='white'),
            #hoverinfo='text',
            hovertemplate="<b>PR Mention: " + event['Display Text'] + "(" + str(event['Date'].date()) + ")<extra></extra>",
            showlegend = False
        ))

        # Add a vertical line for visual aid
        #st.write("date being passed",event_date)
        fig.add_vline(x=event['Date'], line=dict(color=fillcolor, width=1, dash="dash"))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color=fillcolor, dash='dot', width=2),
        name='Web & Blog'
    ))

twt_df['Display Text'] = twt_df['username'] + " - " + twt_df['text'].str[:10] + "..."
twt_grouped = twt_df.groupby(twt_df['created_at'].dt.date).agg({
    'Display Text': lambda x: ' | '.join(x)  # Aggregate display texts
}).reset_index()
twt_grouped['created_at'] = pd.to_datetime(twt_grouped['created_at'])

over_limit = False
if len(twt_grouped) > 30:
    over_limit = True
    twt_grouped = twt_grouped.iloc[np.linspace(0, len(twt_grouped) - 1, 30, dtype=int)]


if selected_options[2]:
    if over_limit:
        st.write("There are more than 30 mentions for Twitter, some mentions will be omitted for better visualization. Select a smaller time frame to display all mentions.")

    color_rgb = colors["Twitter"]
    fillcolor = f'rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},1)'
    for _, event in twt_grouped.iterrows():
        #event_date = event['Date'].timestamp() * 1000
        fig.add_trace(go.Scatter(
            x=[event['created_at']],
            y=[filtered_data['total_page_views'].max() * 1.1],  # slightly above the max value for visibility
            mode='markers',
            marker=dict(size=0, color='white'),
            #hoverinfo='text',
            hovertemplate="<b>PR Mention: " + event['Display Text'] + "(" + str(event['created_at'].date()) + ")<extra></extra>",
            showlegend = False
        ))

        # Add a vertical line for visual aid
        #st.write("date being passed",event_date)
        fig.add_vline(x=event['created_at'], line=dict(color=fillcolor, width=1, dash="dash"))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color=fillcolor, dash='dot', width=2),
        name='Twitter'
    ))


def extract_account_name(url):
    parsed_url = urlparse(url)
    path_segments = parsed_url.path.strip('/').split('/')
    if path_segments:
        return path_segments[0]
    return "" 

fb_df['Account'] = fb_df['Link'].apply(extract_account_name)
fb_df['Display Text'] = fb_df['Account'] + " - " + fb_df['Text'].str[:10] + "..."
fb_grouped = fb_df.groupby(fb_df['Date'].dt.date).agg({
    'Display Text': lambda x: ' | '.join(x)  # Aggregate display texts
}).reset_index()
fb_grouped['Date'] = pd.to_datetime(fb_grouped['Date'])

over_limit = False
if len(fb_grouped) > 30:
    over_limit = True
    fb_grouped = fb_grouped.iloc[np.linspace(0, len(fb_grouped) - 1, 30, dtype=int)]

if selected_options[3]:
    if over_limit:
        st.write("There are more than 30 mentions for Facebook, some mentions will be omitted for better visualization. Select a smaller time frame to display all mentions.")

    color_rgb = colors["Facebook"]
    fillcolor = f'rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},1)'
    for _, event in fb_grouped.iterrows():
        #event_date = event['Date'].timestamp() * 1000
        fig.add_trace(go.Scatter(
            x=[event['Date']],
            y=[filtered_data['total_page_views'].max() * 1.1],  # slightly above the max value for visibility
            mode='markers',
            marker=dict(size=0, color='white'),
            #hoverinfo='text',
            hovertemplate="<b>PR Mention: " + event['Display Text'] + "(" + str(event['Date'].date()) + ")<extra></extra>",
            showlegend = False
        ))

        # Add a vertical line for visual aid
        #st.write("date being passed",event_date)
        fig.add_vline(x=event['Date'], line=dict(color=fillcolor, width=1, dash="dash"))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color=fillcolor, dash='dot', width=2),
        name='Facebook'
    ))

def determine_display_text(row):
    if row['Caption'] == '-':
        return row['Link']  # Use the link if caption is just a dash
    else:
        return row['Caption'][:10] + "..."  # Otherwise, use the caption

if len(ig_df)>0:
    ig_df['Display Text'] = ig_df.apply(determine_display_text, axis=1)
    ig_grouped = ig_df.groupby(ig_df['Date'].dt.date).agg({
        'Display Text': lambda x: ' | '.join(x)  # Aggregate display texts
    }).reset_index()
    ig_grouped['Date'] = pd.to_datetime(ig_grouped['Date'])
else:
    ig_grouped = pd.DataFrame()

over_limit = False
if len(ig_grouped) > 30:
    over_limit = True
    ig_grouped = ig_grouped.iloc[np.linspace(0, len(ig_grouped) - 1, 30, dtype=int)]

if selected_options[4]:
    if over_limit:
        st.write("There are more than 30 mentions for Instagram, some mentions will be omitted for better visualization. Select a smaller time frame to display all mentions.")
    color_rgb = colors["Instagram"]
    fillcolor = f'rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},1)'
    for _, event in ig_grouped.iterrows():
        #event_date = event['Date'].timestamp() * 1000
        fig.add_trace(go.Scatter(
            x=[event['Date']],
            y=[filtered_data['total_page_views'].max() * 1.1],  # slightly above the max value for visibility
            mode='markers',
            marker=dict(size=0, color='white'),
            #hoverinfo='text',
            hovertemplate="<b>PR Mention: " + event['Display Text'] + "(" + str(event['Date'].date()) + ")<extra></extra>",
            showlegend = False
        ))

        # Add a vertical line for visual aid
        #st.write("date being passed",event_date)
        fig.add_vline(x=event['Date'], line=dict(color=fillcolor, width=1, dash="dash"))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color=fillcolor, dash='dot', width=2),
        name='Instagram'
    ))


yt_df['Display Text'] = yt_df['Video Title'].str[:10] + "..."
yt_grouped = yt_df.groupby(yt_df['Date'].dt.date).agg({
    'Display Text': lambda x: ' | '.join(x)  # Aggregate display texts
}).reset_index()
yt_grouped['Date'] = pd.to_datetime(yt_grouped['Date'])

over_limit = False
if len(yt_grouped) > 30:
    over_limit = True
    yt_grouped = yt_grouped.iloc[np.linspace(0, len(yt_grouped) - 1, 30, dtype=int)]

if selected_options[5]:
    if over_limit:
        st.write("There are more than 30 mentions for Youtube, some mentions will be omitted for better visualization. Select a smaller time frame to display all mentions.")
    color_rgb = colors["Youtube"]
    fillcolor = f'rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},1)'
    for _, event in yt_grouped.iterrows():
        #event_date = event['Date'].timestamp() * 1000
        fig.add_trace(go.Scatter(
            x=[event['Date']],
            y=[filtered_data['total_page_views'].max() * 1.1],  # slightly above the max value for visibility
            mode='markers',
            marker=dict(size=0, color='white'),
            #hoverinfo='text',
            hovertemplate="<b>PR Mention: " + event['Display Text'] + "(" + str(event['Date'].date()) + ")<extra></extra>",
            showlegend = False
        ))

        # Add a vertical line for visual aid
        #st.write("date being passed",event_date)
        fig.add_vline(x=event['Date'], line=dict(color=fillcolor, width=1, dash="dash"))

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color=fillcolor, dash='dot', width=2),
        name='Youtube'
    ))



fig.update_layout(hoverdistance=hoverdistance)
graph_placeholder.plotly_chart(fig, use_container_width=True)





