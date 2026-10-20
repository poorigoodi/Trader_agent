# Roadmap — AI Crypto Trading Agent (LBank / Toobit)

> نسخه‌ی اصلاح‌شده: فازهای تکراری حذف شده، فاز Market-Data (خواندنی) از Exchange-Integration
> (حساب‌کاربری/سفارش) جدا شده، و زمان‌بندی‌ها واقع‌بینانه (بر اساس یک توسعه‌دهنده‌ی تنها، پاره‌وقت
> ~۱۰-۱۵ ساعت در هفته، مگر خلاف آن ذکر شود) تنظیم شده‌اند.

## اصول کلی حاکم بر این Roadmap

- **MVP اول، Multi-Agent بعداً.** فازهای ۰ تا ۷ همه در یک orchestrator تک‌فرآیندی پیاده می‌شوند.
- **Real-Time فقط برای داده/ریسک، نه برای LLM.** فراخوانی‌های LLM (فاز ۳ به بعد) batched/periodic‌اند.
- **Backtest و Live ممکن است منبع داده‌ی متفاوت داشته باشند** — این باید همیشه صریح مستند شود.
- **هیچ فازی وارد Live Trading واقعی نمی‌شود مگر فازهای ۰ تا ۸ کامل و معیارهای عبور برآورده شده باشند.**

---

## جدول فازها

| فاز | هدف | کارهای کلیدی | ماژول/فایل اصلی | تکنولوژی | خروجی قابل‌تحویل | معیار موفقیت | وابستگی | سختی | زمان تقریبی |
|---|---|---|---|---|---|---|---|---|---|
| **0 — Research & Architecture** | مستندسازی معماری و تصمیمات کلیدی | بررسی مستندات ccxt برای LBank/Toobit؛ تعیین scope دقیق MVP؛ نوشتن سند معماری | `docs/architecture.md` | Markdown | سند معماری تأییدشده | تیم/خودت روی معماری توافق دارید | — | کم | ۲-۳ روز |
| **1 — Market Data (Read-Only)** | دریافت پایدار داده‌ی بازار | اتصال ccxt به LBank/Toobit (public endpoints)، OHLCV چند تایم‌فریمه، ticker، order book پایه، ذخیره در DB | `market/data_feed.py`, `database/models.py` | Python, ccxt, SQLite/Postgres | اسکریپتی که به‌صورت زنده کندل BTC/ETH را می‌گیرد و ذخیره می‌کند | داده بدون قطعی/تکرار برای ≥۲۴ ساعت پیوسته جمع می‌شود | فاز ۰ | کم-متوسط | ۳-۵ روز |
| **2 — Technical Analysis Engine** | تولید اندیکاتور و ساختار بازار | EMA/SMA/RSI/MACD/ATR/Bollinger/VWAP؛ تشخیص ساده‌ی Trend و Support/Resistance | `indicators/*.py`, `market_structure/*.py` | pandas / pandas-ta | ماژول اندیکاتور با تست واحد | خروجی اندیکاتورها روی داده‌ی تست دستی Cross-check شده | فاز ۱ | متوسط | ۴-۶ روز |
| **3 — Rule-Based Signal + Risk (MVP Core)** | تولید سیگنال Deterministic و مدیریت ریسک | استراتژی Rule-Based ساده؛ Position Sizing؛ SL/TP/R:R؛ Invalidation Conditions | `strategies/*.py`, `risk/*.py` | Python خالص (Deterministic) | سیگنال LONG/SHORT/NO_TRADE با دلیل و اعداد ریسک | سیگنال‌ها روی داده‌ی تاریخی به‌صورت دستی قابل‌تأیید‌اند | فاز ۲ | متوسط | ۵-۷ روز |
| **4 — Paper Trading Engine** | شبیه‌سازی معامله بدون پول واقعی | کیف‌پول مجازی، شبیه‌سازی سفارش/کارمزد/اسلیپیج، PnL، Trade Journal | `paper_trading/*.py` | SQLite/Postgres | چرخه‌ی کامل: سیگنال → Paper Trade → PnL ثبت‌شده | حداقل ۲ هفته اجرای پیوسته بدون کرش | فاز ۳ | متوسط | ۴-۶ روز |
| **5 — Backtesting Engine** | ارزیابی استراتژی روی داده‌ی تاریخی | Backtest vectorized/event-driven؛ داده‌ی تاریخی از منبع مجزا (نه لزوماً LBank/Toobit)؛ جلوگیری از Look-Ahead/Leakage؛ محاسبه‌ی Slippage/Fee واقعی | `backtesting/engine.py`, `backtesting/metrics.py` | pandas/vectorbt (اختیاری) | گزارش Backtest با ROI, Win Rate, Profit Factor, Sharpe, Sortino, Max DD, Expectancy | نتایج با Paper Trading هم‌جهت‌اند (نه لزوماً برابر) | فاز ۴ (منطق مشترک) | متوسط-بالا | ۷-۱۰ روز |
| **6 — News Intelligence (پایه)** | جمع‌آوری خبر رایگان/کم‌هزینه | RSS خبرگزاری‌های کریپتو، CryptoPanic (free tier)، طبقه‌بندی ساده‌ی خبر (مهم/کم‌اهمیت) | `news/collector.py`, `news/classifier.py` | Python, RSS, یک مدل کوچک طبقه‌بندی یا Rule-Based | فید خبر ذخیره‌شده و برچسب‌خورده | خبرهای مهم (CPI/Fed/Hack/Listing) به‌موقع (<۵ دقیقه تأخیر) ثبت می‌شوند | فاز ۱ (موازی با ۲-۵) | متوسط | ۵-۷ روز |
| **7 — LLM Decision Layer** | افزودن ریزنینگ LLM روی سیگنال Rule-Based | پرامپت خبر+سنتیمنت+تکنیکال → توضیح/تأیید یا رد سیگنال؛ محاسبه‌ی Confidence کالیبره (نه خام LLM)؛ ذخیره‌ی کامل Prompt/Response | `ai_decision/llm_layer.py` | Claude API | سیگنال نهایی با Reasoning متنی و Confidence عددی | خروجی LLM هرگز مستقیم سفارش نمی‌سازد؛ همیشه از Validation Layer رد می‌شود | فاز ۳, ۶ | متوسط-بالا | ۶-۸ روز |
| **8 — Exchange Integration (Trade/Account API)** | آماده‌سازی اجرای واقعی (هنوز بدون پول واقعی) | `ExchangeInterface` + `LBankAdapter`/`ToobitAdapter` روی ccxt؛ Auth؛ Order Placement/Cancel در حالت Sandbox یا حجم صفر؛ مدیریت امن کلید | `exchanges/*.py` | ccxt, Secrets Manager | امکان ارسال سفارش تستی با حجم بسیار کوچک | سفارش تستی با موفقیت ثبت/لغو می‌شود؛ کلید بدون مجوز Withdrawal | فاز ۳-۵ کامل باشند | بالا | ۶-۸ روز |
| **9 — Live Trading Safety + Go-Live کوچک** | فعال‌سازی معامله‌ی واقعی با سرمایه‌ی کوچک | Kill Switch, Max Daily Loss, Circuit Breaker, Duplicate Order Protection, Position Reconciliation | `execution/*.py`, `safety/*.py` | Python | حالت Live با سرمایه‌ی کوچک و محدودیت سخت | حداقل ۴ هفته Paper Trading با نتایج قابل‌قبول قبل از این فاز | فاز ۸ + معیارهای عبور از فاز Paper Trading | بالا | ۷-۱۰ روز |
| **10 — Optimization & Scaling** | بهبود مستمر و مقیاس‌پذیری | Multi-Agent واقعی (اختیاری)، مدل‌های ML، چند صرافی/چند دارایی، Continuous Learning | متنوع | متنوع | نسخه‌ی بالغ‌تر سیستم | تصمیم‌گیری بر اساس داده‌ی واقعی عملکرد | فاز ۹ پایدار | بالا | مستمر / پروژه‌ی جدا |

> **مجموع زمان تخمینی تا پایان فاز ۹ (بدون فاز ۱۰):** حدود ۳ تا ۴ ماه کاری پاره‌وقت برای یک
> توسعه‌دهنده‌ی تنها — نه چند هفته. اگر تمام‌وقت روی آن کار کنی، می‌توان به حدود ۶-۸ هفته فشرده کرد،
> اما کیفیت تست و Backtesting معمولاً قربانی عجله می‌شود؛ پیشنهاد نمی‌شود.

---

## معیارهای عبور بین حالت‌ها (Human-in-the-Loop)

| از حالت | به حالت | معیار عبور |
|---|---|---|
| Analysis Only (فاز ۲-۳) | Signal Mode (فاز ۴-۷) | سیگنال‌ها به‌صورت دستی روی ≥۵۰ کندل تاریخی معقول به‌نظر می‌رسند |
| Signal Mode | Paper Trading فعال (فاز ۴-۸) | Backtest حداقل ۶ ماه داده با Profit Factor > ۱ و Max Drawdown قابل‌قبول (مثلاً <۲۵٪) |
| Paper Trading | Live با سرمایه‌ی کوچک (فاز ۹) | حداقل ۴ هفته Paper Trading پیوسته، Win Rate/Profit Factor پایدار (نه فقط یک دوره‌ی خوش‌شانس)، صفر باگ بحرانی در Execution/Risk، تأیید صریح کاربر |
| Live کوچک | Live با سرمایه‌ی بیشتر | حداقل ۴-۸ هفته‌ی دیگر با نتایج پایدار روی پول واقعی + بدون فعال‌شدن غیرمنتظره‌ی Kill Switch |

---

## MVP Checklist (فازهای ۰ تا ۴)

- [ ] اتصال ccxt به LBank (یا Binance برای تست توسعه) برای BTC/USDT و ETH/USDT برقرار است
- [ ] کندل ۱۵ دقیقه‌ای به‌صورت زنده و بدون افت داده جمع‌آوری می‌شود
- [ ] اندیکاتورهای EMA, RSI, MACD, ATR, Bollinger با تست واحد پیاده شده‌اند
- [ ] استراتژی Rule-Based سیگنال LONG/SHORT/NO_TRADE با دلیل تولید می‌کند
- [ ] هر سیگنال شامل Entry, SL, TP, R:R, Position Size, Invalidation Condition است
- [ ] Position Sizing و SL/TP کاملاً Deterministic (بدون LLM) هستند
- [ ] موتور Paper Trading کیف‌پول مجازی، PnL و Trade Journal دارد
- [ ] همه‌چیز (کندل، سیگنال، Trade) در دیتابیس محلی لاگ می‌شود
- [ ] هیچ کلید API واقعی و هیچ مسیر سفارش واقعی در کد نیست
- [ ] README مشخص می‌کند چه‌چیزی *عمداً* در MVP نیست: LLM, News, Sentiment, Multi-Agent, Live Trading

**عمداً در MVP نیست:** News Intelligence، Sentiment Analysis، LLM Reasoning، معماری Multi-Agent،
Backtesting کامل با News تاریخی، اتصال به بیش از ۲ نماد، هرگونه اجرای سفارش واقعی.

---

## برنامه‌ی ۷ روز اول (فقط اسکلت MVP — نه سیستم کامل)

| روز | کار |
|---|---|
| ۱ | راه‌اندازی پروژه (venv, requirements, ساختار پوشه، Git)؛ نوشتن `docs/architecture.md` کوتاه (فاز ۰) |
| ۲ | اتصال ccxt به LBank/Binance؛ دریافت و ذخیره‌ی OHLCV تاریخی و زنده برای BTC/USDT، ETH/USDT |
| ۳ | پیاده‌سازی اندیکاتورها (EMA, RSI, ATR, MACD, Bollinger) + تست واحد هرکدام |
| ۴ | پیاده‌سازی استراتژی Rule-Based ساده (مثلاً EMA Cross + RSI Filter) با خروجی LONG/SHORT/NO_TRADE + دلیل |
| ۵ | پیاده‌سازی Risk Module: Position Sizing (٪ ثابت از موجودی)، محاسبه‌ی SL/TP/R:R |
| ۶ | پیاده‌سازی موتور Paper Trading (کیف‌پول مجازی، شبیه‌سازی سفارش با کارمزد/اسلیپیج، PnL) |
| ۷ | اتصال همه‌چیز در `main.py` (حلقه‌ی زمان‌بندی‌شده)، اجرای واقعی روی داده‌ی زنده برای چند ساعت، بررسی لاگ‌ها، نوشتن README |

بعد از این ۷ روز، فاز ۵ (Backtesting) و فاز ۶ (News) شروع می‌شوند — این دو معمولاً هرکدام ۱-۲ هفته‌ی
دیگر می‌برند.

---

## اولین Coding Task (از اینجا شروع کن)

**فایل اول:** `app/market/data_feed.py`

**کار:** یک کلاس `MarketDataFeed` بنویس که:
1. با `ccxt` به Exchange مشخص‌شده (پارامتر ورودی، پیش‌فرض برای تست: `binance`، هدف نهایی: `lbank`)
   وصل شود.
2. متد `fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame` داشته باشد که ستون‌های
   `timestamp, open, high, low, close, volume` را برگرداند.
3. متد `stream_latest(symbol, timeframe, callback)` یا معادل polling‌ی (اگر WebSocket فعلاً پیچیده
   است، Polling هر ۳۰-۶۰ ثانیه کافی‌ست برای شروع) که کندل جدید را به یک تابع callback بدهد.
4. خطاها (قطعی شبکه، Rate Limit) را با Retry با Backoff مدیریت کند، نه Crash.
5. یک تست واحد ساده بنویس که با داده‌ی Mock (نه اتصال واقعی) بررسی کند DataFrame خروجی ساختار درستی
   دارد.

این فایل پایه‌ی همه‌ی فازهای بعدی است؛ تا این‌جا Deterministic و کاملاً بدون LLM/کلید خصوصی می‌ماند.
