# Dataset Description

The dataset (`news_with_stock_labels_devided.csv`) consists of the following columns:

- **Date**: The date of the recorded stock market change.
- **Label**:
  - `up`: Indicates that the Dow Jones index has risen or remained stable.
  - `down`: Indicates that the Dow Jones index has fallen.
- **Change**: Represents the actual change in the Dow Jones index, which can be a positive number (indicating an increase) or a negative number (indicating a decrease).
- **News Headlines by Category**:
  - **Tech_1, Tech_2, Tech_3, Tech_4**: News headlines related to technology.
  - **Business_1, Business_2, Business_3, Business_4**: News headlines related to business.
  - **Market_1, Market_2, Market_3, Market_4**: News headlines related to market trends.
  - **Economy_1, Economy_2, Economy_3, Economy_4**: News headlines related to economic indicators.
  - **Events_1, Events_2, Events_3, Events_4**: News headlines related to major world events.
- **News Sources**:
  - **Tech_1_resource, Tech_2_resource, ..., Events_4_resource**: Indicate the sources of the corresponding news headlines. Each value represents the media outlet, website, or publication from which the respective headline was obtained.

This information helps assess the credibility and potential biases of different sources, allowing for a more comprehensive analysis of how various news providers impact market sentiment.

## Additional Datasets

In addition to this dataset, there are three other datasets:

1. **stock_data.csv**: This dataset contains stock market data, including the date, the change in the Dow Jones index, and the label indicating whether the market went up or down on that specific date. It provides the core stock market information necessary for analyzing market trends.

2. **news_data.csv**: This dataset includes daily news headlines from various categories, such as technology, business, market trends, economy, and major events. Each category has multiple headlines, offering a detailed view of the news landscape on a given day, which can be cross-referenced with stock market movements.

3. **news_data_with_labels.csv**: This dataset combines the news headlines with the stock market changes and labels. It links the news content directly to the corresponding market performance, providing a more comprehensive dataset for analysis of how specific news topics and headlines influence market behavior.

Together, these datasets provide a multi-faceted approach to understanding the relationship between news and stock market performance.
