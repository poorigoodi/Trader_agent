# پرامپت اصلی طراحی AI Trading Agent — نسخه اصلاح‌شده (بدون باگ)

> این فایل نسخه بازبینی‌شده‌ی پرامپت اولیه‌ست. تکرارهای داخلی حذف شده، تناقض‌های معماری رفع شده،
> و ۱۳ محدودیت/قانون جدید (بخش زیر) اضافه شده تا خروجی نهایی واقع‌بینانه و قابل‌اجرا باشد.
> این فایل را می‌توانی مستقیماً به‌عنوان یک پرامپت کامل به Claude (یا هر مدل دیگر) بدهی تا Roadmap و
> معماری کامل را از نو تولید کند، یا به‌عنوان سند مرجع پروژه نگه داری.

---

## نقش (Role)

تو یک **Senior AI Agent Architect + Quant Developer + Crypto Trading Systems Engineer** هستی با تجربه در
طراحی سیستم‌های معاملاتی الگوریتمی، LLM Agents، تحلیل تکنیکال، تحلیل اخبار و Sentiment Analysis،
مدیریت ریسک، Backtesting و اتصال به API صرافی‌های ارز دیجیتال.

هدف: طراحی و مستندسازی یک **AI Trading Agent برای بازار کریپتو** که نهایتاً روی صرافی‌های **LBank** و
**Toobit** کار کند؛ ابتدا در حالت Paper Trading و در صورت تأیید کاربر، در حالت Live.

---

## قوانین پایه و محدودیت‌های اجباری (Ground Rules) — قبل از هر چیز باید رعایت شوند

این بخش نتیجه‌ی بازبینی نسخه‌ی قبلی پرامپت است و باگ‌های شناسایی‌شده را می‌بندد. هر پاسخی که این
قوانین را نقض کند، ناقص محسوب می‌شود.

1. **MVP ≠ معماری Multi-Agent کامل.**
   نسخه‌ی اول (MVP) باید یک orchestrator تک‌فرآیندی با ماژول‌های داخلی جدا (نه سرویس‌های مستقل با
   Message Queue) باشد. تفکیک واقعی به Agentهای مستقل، فاز جداگانه‌ای *بعد از* اثبات MVP است.

2. **Real-Time فقط برای لایه‌های deterministic.**
   دریافت داده، محاسبه‌ی اندیکاتور و بررسی ریسک باید نزدیک به Real-Time (ثانیه‌ای) باشند.
   فراخوانی LLM (تحلیل خبر، سنتیمنت، ریزنینگ نهایی) باید batched/periodic باشد — مثلاً روی بسته‌شدن
   کندل یا هر N دقیقه — نه روی هر تیک قیمت. این تفکیک باید در معماری صریحاً مشخص شود.

3. **جداسازی Market-Data API از Trade/Account API.**
   در Exchange Abstraction Layer، متدهای عمومی/فقط-خواندنی (kline، order book، ticker، funding rate)
   باید از متدهای حساب‌کاربری/سفارش (balance، place order، positions) در یک اینترفیس، اما دو دسته‌ی
   منطقی جدا تعریف شوند؛ چون فاز Market Data فقط به دسته‌ی اول نیاز دارد و نباید منتظر پیاده‌سازی
   کامل Trade API بماند.

4. **منبع داده‌ی Backtest ممکن است با صرافی اجرا فرق داشته باشد — این باید صریح مدیریت شود.**
   LBank و Toobit عمق تاریخی کندل نامشخص/محدودی از طریق API عمومی ارائه می‌دهند (مستندات رسمی Toobit
   حتی به این محدودیت اشاره‌ای نکرده است). بنابراین داده‌ی Backtest احتمالاً باید از یک منبع مجزا و
   قابل‌اعتماد (مثلاً داده‌ی عمومی Binance، یا یک Data Vendor) تأمین شود، درحالی‌که اجرای زنده روی
   LBank/Toobit انجام می‌شود. سیستم باید **Basis Risk** بین این دو منبع را صریحاً گزارش کند، نه فرض
   کند قیمت‌ها یکسان‌اند.

5. **ریسک اجرای واقعی روی LBank/Toobit را دست‌کم نگیر.**
   داده‌ی واقعی (CoinGecko، شهریور ۲۰۲۶): میانگین اسپرد خرید-فروش LBank ≈ ۰.۴۴٪ و Toobit ≈ ۰.۵۴٪
   (در برابر معمولاً زیر ۰.۰۲٪ در صرافی‌های بزرگ)؛ رتبه‌ی امنیت سایبری CORE3: LBank = ۲ از ۵،
   Toobit = ۳ از ۵. بنابراین فرض Slippage و هزینه در Backtest و Position Sizing باید محافظه‌کارانه
   و قابل‌تنظیم باشد (پارامتر، نه عدد ثابت خوش‌بینانه)، و ریسک امنیتی صرافی باید در سیاست نگهداری
   موجودی (چقدر روی صرافی بماند در برابر چقدر جابه‌جا شود) لحاظ شود.

6. **Confidence Score باید کالیبره باشد، نه احساس LLM.**
   عدد Confidence هر Trade نباید صرفاً خروجی خودارزیابی LLM باشد. باید از یک تابع امتیازدهی
   deterministic محاسبه شود که ترکیبی از توافق سیگنال‌های تکنیکال/خبر/سنتیمنت + نرخ موفقیت تاریخی
   ستاپ‌های مشابه (از نتایج Backtest) است.

7. **جلوگیری از نشتی دانش LLM در Backtest (یک نوع نامرئی از Look-Ahead Bias).**
   وقتی خبر تاریخی به LLM داده می‌شود، این ریسک وجود دارد که مدل از دانش آموزشی خودش نتیجه‌ی بعدی
   رویداد را از قبل «بداند». این محدودیت باید در گزارش Backtest به‌صراحت به‌عنوان یک ریسک شناخته‌شده
   ثبت شود و در حد امکان با محدود کردن اطلاعات ورودی LLM به فقط متن خبر (بدون بازه‌ی زمانی گسترده یا
   اشاره به نتیجه) کاهش یابد.

8. **بودجه‌ی داده مشخص باشد.**
   منابع خبر/سنتیمنت باید بر اساس هزینه دسته‌بندی شوند (رایگان/ارزان در برابر پولی مانند X/Twitter API
   یا LunarCrush). MVP باید فقط از منابع رایگان/کم‌هزینه استفاده کند (RSS خبرگزاری‌های کریپتو،
   CryptoPanic free tier، Alternative.me Fear & Greed Index).

9. **امنیت کلید API فراتر از «امن نگه‌دار».**
   کلید API صرافی باید: مجوز Withdrawal غیرفعال داشته باشد؛ در صورت پشتیبانی صرافی، IP-Whitelist شود؛
   در Secrets Manager یا فایل رمزنگاری‌شده نگهداری شود (نه .env ساده‌ی متنی)؛ و چرخش دوره‌ای داشته باشد.
   Agent (لایه‌ی LLM) هرگز نباید مستقیماً به مقدار خام Secret دسترسی داشته باشد.

10. **زمان‌بندی واقع‌بینانه.**
    این یک سیستم Production-grade کامل است که برای یک توسعه‌دهنده‌ی تنها معمولاً **چند ماه** طول
    می‌کشد، نه چند هفته. «۷ روز اول» فقط باید به یک اسکلت حداقلی برسد (داده + اندیکاتور + سیگنال
    Rule-Based ساده + ثبت Paper Trade)، نه به سیستم کامل.

11. **Correlation Risk در MVP کم‌اهمیت است.**
    چون MVP فقط BTC/ETH دارد (همبستگی معمولاً بالای ۰.۸)، این ماژول باید ساده/Placeholder باشد و فقط
    از فاز Multi-Asset به بعد جدی پیاده‌سازی شود.

12. **Observability باید شامل خام Prompt/Response باشد.**
    برای هر تصمیم LLM: متن دقیق Prompt ارسالی، شناسه/نسخه‌ی مدل، پارامترها (temperature و...)، و
    Response خام باید ذخیره شوند — نه فقط خلاصه‌ی Reasoning.

13. **JSON Schema باید چند-صرافی/چند-بازار باشد.**
    فیلدهای `exchange`، `market_type` (spot/futures)، `leverage` (فقط برای futures، در spot باید
    null باشد)، `take_profit` به‌صورت آرایه‌ای از `{price, size_pct}` برای خروج پله‌ای، و
    `client_order_id` برای جلوگیری از Duplicate Order باید در Schema باشند.

14. **Exchange Abstraction روی ccxt بنا شود، نه از صفر.**
    کتابخانه‌ی `ccxt` هم‌اکنون از هر دو صرافی LBank و Toobit (شامل Spot و USDT-M Futures، و
    WebSocket/Pro) پشتیبانی می‌کند. به‌جای نوشتن REST Client از صفر، `LBankAdapter` و `ToobitAdapter`
    باید Wrapperهای نازکی روی ccxt باشند که اینترفیس داخلی `ExchangeInterface` را پیاده‌سازی می‌کنند؛
    فقط برای قابلیت‌هایی که ccxt پوشش نمی‌دهد (اگر بود) باید کد اختصاصی نوشته شود.

15. **بدون تکرار محتوا.**
    Tech Stack فقط یک‌بار (به‌صورت جدول)، Agent Architecture فقط یک‌بار (به‌صورت جدول)، و
    Development Roadmap فقط یک‌بار (به‌صورت جدول فاز/کار/خروجی/معیار/وابستگی/سختی/زمان) در پاسخ نهایی
    بیاید. بخش خلاصه‌ی پایانی («نتیجه نهایی») فقط باید ارجاع/فشرده‌سازیِ همان جداول باشد، نه تولید
    دوباره‌ی محتوای کامل.

---

## بخش‌های موردنیاز (تعریف نهایی، بدون تکرار)

### ۱. تعریف دقیق سیستم
اجزای Agent، وظایف هرکدام، کدام بخش‌ها LLM/ML/Rule-Based/Deterministic باشند، کدام بخش‌ها Real-Time
هستند (طبق قانون ۲). یک معماری پیشنهادی سطح‌بالا ارائه شود.

### ۲. معماری کلی
Data Layer (OHLCV, Order Book, Trades, Funding Rate, Open Interest, Liquidations, Volume, Volatility,
Market Cap, BTC Dominance — با ذکر این‌که Market Cap/Dominance معمولاً از یک Aggregator جدا مثل
CoinGecko/CMC می‌آید، نه از خود صرافی) → Technical Analysis Engine → News Intelligence →
Sentiment Engine → AI Decision Engine.

### ۳. طراحی Agent (Single یا Multi-Agent)
مقایسه و انتخاب نهایی، با رعایت قانون ۱ (MVP تک‌فرآیندی، تفکیک واقعی بعداً). جدول Agentها فقط یک‌بار
در «نتیجه نهایی» بیاید: وظیفه / Input / Output / ابزار / مدل پیشنهادی / نیاز به LLM / ارتباط با
Agentهای دیگر.

### ۴. Trading Decision Pipeline
Market Data → Data Validation → Technical Analysis → News Analysis → Sentiment Analysis →
Market Regime Detection → Strategy Evaluation → Risk Management → Trade Scoring (طبق قانون ۶) →
Final Decision → Paper Trade → Execution. هر مرحله دقیقاً چه‌کاری انجام می‌دهد و کدام بخش‌ها
Async/Batched هستند (طبق قانون ۲) مشخص شود.

### ۵. خروجی استاندارد Trade (JSON Schema)
طبق قانون ۱۳ و با ذکر Invalidation Conditions، Technical/News/Sentiment Reasoning جدا.

### ۶. Risk Management (Deterministic، نه LLM)
Max Risk/Trade, Max Daily Loss, Max Drawdown, Position Sizing, SL/TP, R:R, Leverage Limits,
Max Open Positions, Correlation Risk (طبق قانون ۱۱)، Exposure, Liquidation Risk, Slippage
(طبق قانون ۵)، Fees. توضیح صریح چرا این لایه نباید به LLM سپرده شود.

### ۷. Backtesting
Historical Data (طبق قانون ۴ — منبع جدا از صرافی اجرا)، تایم‌فریم‌ها، ورود News تاریخی
(طبق قانون ۷ — ریسک نشتی دانش LLM)، جلوگیری از Look-Ahead Bias و Data Leakage، Slippage/Fee/Funding
واقعی (طبق قانون ۵)، Walk-Forward و Out-of-Sample Testing. Metrics: ROI, Win Rate, Profit Factor,
Sharpe, Sortino, Max Drawdown, Expectancy, Avg R:R, CAGR, Number of Trades.

### ۸. Paper Trading
Virtual Wallet/Positions، Order/Fee/Slippage Simulation، PnL، Drawdown، Trade Journal، و معیارهای
عبور به Live (حداقل تعداد Trade، حداقل دوره‌ی زمانی، حداقل Profit Factor و Max Drawdown قابل‌قبول).

### ۹. اتصال به LBank و Toobit
REST/WebSocket، Market/Account Data، Order Placement/Cancel، Futures/Spot، Authentication،
Rate Limits — بر پایه‌ی ccxt (طبق قانون ۱۴). `ExchangeInterface` → `LBankAdapter` / `ToobitAdapter`.
امنیت کلید طبق قانون ۹.

### ۱۰. Live Trading Safety
Kill Switch, Max Position Size, Max Daily Loss, Max Order Value, API Permission Restrictions,
Emergency Close, Circuit Breaker, Duplicate Order Protection (با `client_order_id`)، Order
Validation, Price Sanity Check, Balance Check, Position Reconciliation, Retry Policy, Rate Limit
Handling.

### ۱۱. انتخاب تکنولوژی (فقط یک جدول — طبق قانون ۱۵)
Component | Technology | Reason — شامل Backend, DB (PostgreSQL/TimescaleDB), Redis, Pandas/Polars,
TA (pandas-ta یا TA-Lib)، ML (بعد از MVP)، LLM (Cloud vs Local)، Agent Framework یا بدون آن،
Docker/Compose. مشخص شود در MVP چه چیزهایی *نباید* اضافه شوند (Kafka، Vector DB، Multi-Agent
Framework، ML models).

### ۱۲. انتخاب مدل AI
News Classification, Sentiment, Reasoning, Market Regime Detection, Trade Decision, Summarization —
و مرز دقیق LLM در برابر ML کلاسیک در برابر Rule-Based (سیستم نباید بیش‌ازحد به LLM وابسته باشد).

### ۱۳. Memory Architecture
Short-Term / Long-Term / Trade / Strategy / Market Regime Memory. Vector DB فقط اگر واقعاً لازم
باشد (نه به‌طور پیش‌فرض در MVP).

### ۱۴. Observability
طبق قانون ۱۲ (خام Prompt/Response). Market Snapshot, Indicators, News Used, Sentiment, Reasoning,
Strategy, Entry/SL/TP, Position Size, Result, PnL, Exit Reason برای هر Trade. Logging/Monitoring/
Alerting.

### ۱۵. جلوگیری از Hallucination (Validation Layer)
خبر جعلی، داده قدیمی، قیمت اشتباه، تصمیم بدون داده کافی، اشتباه Symbol/Position Size/Leverage،
Duplicate Trade، تحلیل متناقض، و Confidence غیرکالیبره (طبق قانون ۶).

### ۱۶. Human-in-the-Loop
Mode 1 (Analysis Only) → Mode 2 (Signal Only) → Mode 3 (Autonomous)، با معیار عبور مشخص برای هر
گذار (حداقل روزهای Paper Trading موفق، حداقل تعداد Trade، تأیید صریح کاربر).

### ۱۷. UI / Dashboard
Current Market, Positions, Signals, Trade History, PnL, Drawdown, Win Rate, Agent Decisions, News,
Sentiment, Risk Exposure, System Health. پیشنهاد ساده برای MVP.

### ۱۸. ساختار پروژه (Project Structure)
یک ساختار پوشه‌ی Python حرفه‌ای متناسب با معماری بالا (جدا‌سازی agents/market/strategies/indicators/
news/sentiment/risk/execution/exchanges/backtesting/paper_trading/portfolio/database/monitoring/api/
config/tests).

### ۱۹. Development Roadmap (فقط یک‌بار — طبق قانون ۱۵)
جدول: Phase | Tasks | Modules/Files | Tech | Deliverable | Success Criteria | Dependency | Difficulty
| Time Estimate (Solo Dev). فازها: Phase 0 (Research & Architecture) تا Phase 10
(Optimization & Scaling)، با رعایت قانون ۳ (جداسازی Market-Data API از Trade API بین فازها) و
قانون ۱۰ (زمان‌بندی واقع‌بینانه).

### ۲۰. MVP
دقیقاً طبق قانون ۱ (تک‌فرآیندی، نه Multi-Agent)، قانون ۸ (فقط داده‌ی رایگان)، قانون ۱۱ (Correlation
ساده). BTC/USDT و ETH/USDT، بدون پول واقعی. مشخص شود چه Featureهایی عمداً حذف می‌شوند.

### ۲۱. مسیر توسعه‌ی آینده
MVP → Better Data → Better Strategies → ML Models → Multi-Agent واقعی → Advanced Backtesting →
Small Capital Live → Multi-Exchange → Portfolio Management → Continuous Learning — با ذکر این‌که
هر مرحله *چه زمانی* واقعاً ارزش اضافه‌شدن دارد.

### ۲۲. اشتباهات رایج
حداقل ۱۵ اشتباه، شامل Overfitting, Data Leakage, Look-Ahead Bias, LLM Hallucination
(و نشتی دانش LLM طبق قانون ۷)، Overtrading, Curve Fitting, Excessive Leverage, Poor Risk Management,
Bad Data, News Latency, Slippage/اسپرد بالا در صرافی‌های کوچک (طبق قانون ۵), API Failures,
Lack of Validation, Confidence غیرکالیبره.

### ۲۳. نتیجه‌ی نهایی (خلاصه، نه تکرار)
1. Architecture Diagram (متنی)
2. جدول Technology Stack (ارجاع به بخش ۱۱، بدون تکرار محتوا)
3. جدول Agent Architecture (ارجاع به بخش ۳)
4. جدول Development Roadmap (ارجاع به بخش ۱۹)
5. MVP Checklist
6. برنامه‌ی ۷ روز اول (فقط اسکلت — طبق قانون ۱۰)
7. اولین Coding Task دقیق: نام فایل/ماژول شروع

---

## قوانین مهم (بدون تغییر نسبت به نسخه‌ی اول)

- از وعده‌ی سوددهی یا Win Rate تضمینی خودداری کن.
- فرض نکن AI الزاماً می‌تواند بازار را پیش‌بینی کند.
- بین Signal Generation و Risk/Execution تفاوت قائل شو.
- Execution و Risk Management باید تا حد ممکن deterministic و قابل‌کنترل باشند.
- LLM نباید مستقیماً و بدون Validation سفارش ارسال کند.
- امنیت API Keyها را جدی بگیر (طبق قانون ۹).
- قبل از Live Trading حتماً Backtesting و Paper Trading در معماری باشد.
- اگر تکنولوژی یا API خاصی تغییر کرده، اطلاعات را بر اساس مستندات فعلی بررسی کن.
- برای LBank و Toobit در صورت نیاز مستندات رسمی API (و مستندات ccxt برای این دو صرافی) را بررسی کن.
- اگر بین چند معماری اختلاف است، مقایسه کن و یک گزینه‌ی اصلی را پیشنهاد بده.
- اولویت با قابلیت اطمینان، کنترل ریسک، تست‌پذیری و مانیتورینگ است، نه پیچیدگی.

پاسخ نهایی باید کاملاً عملی، مهندسی‌شده، بدون تکرار محتوا، و قابل‌اجرا باشد.
