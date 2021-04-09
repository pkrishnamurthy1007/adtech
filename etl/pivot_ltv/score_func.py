def predict_duration_percent(plan_length_group, age, sold_month, has_vision=False, has_children=False):
    """
        for all STM / BTM plans 1 year or less

        plan_type: int, one of [90, 180, 364] (based on sold duration, 180x1 and 90x2 would both be 180)
        age: int, primary insured age at time of sale
        sold_month int, month during the year the plan was purchased
        has_vision: true/false if vision is an addon
        has_children true/false if a child (18 or under) is on the plan

        output: portion of sold duration expected before cancellation.

        output * sold_duration_days = duration_inforce

    """
    output = None
    if plan_length_group == 90:
        """
        primary_age               0.0008
        has_children             -0.0189
        month_group_90_1_8_7_3   -0.0109
        month_group_90_2_6_5     -0.0342
        intercept                 0.8455
        """
        output = 0.8455
        output += (age * 0.0008)
        if has_children: output += -0.0189
        if sold_month in [1, 3, 7, 8]: output += -0.0109
        if sold_month in [2, 5, 6]: output += -0.0342

    if plan_length_group == 90:
        """
        has_vision                -0.0849
        primary_age                0.0018
        has_children              -0.0427
        month_group_180_2_1_7      0.1244
        month_group_180_3_4_5_6    0.1635
        month_group_180_8_9_12     0.0813
        intercept                  0.5645
        """
        output = 0.5645
        output += (age * 0.0018)
        if has_children: output += -0.0427
        if has_vision: output += -0.0849
        if sold_month in [1, 2, 7]: output += 0.1244
        if sold_month in [3, 4, 5, 6]: output += 0.1635
        if sold_month in [8, 9, 12]: output += 0.0813

    if plan_length_group == 364:
        """
        has_vision               -0.0514
        primary_age               0.0021
        has_children             -0.0237
        month_group_364_2_3_10   -0.0642
        month_group_364_4_5_6    -0.1189
        month_group_364_7_8_9    -0.1665
        intercept                 0.5258
        dtype: float64
        """
        output = 0.5258
        output += (age * 0.0021)
        if has_children: output += -0.0237
        if has_vision: output += -0.0514
        if sold_month in [2, 3, 10]: output += -0.0642
        if sold_month in [4, 5, 6]: output += -0.1189
        if sold_month in [7, 8, 9]: output += -0.1665

    return output