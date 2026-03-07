from fredapi import Fred

fred = Fred(api_key="95ce06c778b5899068cd2cebe98023fd ")
gdp = fred.get_series("GDP")
