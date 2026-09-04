function lambda = arrival_rate_per_s(cfg)
%ARRIVAL_RATE_PER_S Mean patient arrival rate from the operating calendar.
%   lambda = annual_patients / (days * hours * 3600). Poisson mean
%   interarrival time is 1/lambda.
    H = cfg.operating_days_per_year * cfg.operating_hours_per_day * 3600;
    lambda = cfg.annual_patients / H;
end
