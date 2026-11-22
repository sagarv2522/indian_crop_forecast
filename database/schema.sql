-- ═══════════════════════════════════════════════════════
-- Crop Price Forecasting Platform — Database Schema
-- Run once on fresh DB: psql -d your_db -f schema.sql
-- ═══════════════════════════════════════════════════════

-- Reference: all unique crops
CREATE TABLE IF NOT EXISTS crops (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(150) NOT NULL,
    variety     VARCHAR(150),
    category    VARCHAR(80),          -- vegetable / cereal / spice / fruit
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, variety)
);

-- Reference: all unique mandi markets
CREATE TABLE IF NOT EXISTS markets (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(200) NOT NULL,
    district    VARCHAR(100),
    state       VARCHAR(100),
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, state)
);

-- Core fact table: daily price records
CREATE TABLE IF NOT EXISTS price_records (
    id            SERIAL PRIMARY KEY,
    crop_id       INTEGER NOT NULL REFERENCES crops(id) ON DELETE CASCADE,
    market_id     INTEGER NOT NULL REFERENCES markets(id) ON DELETE CASCADE,
    arrival_date  DATE NOT NULL,
    min_price     DECIMAL(10, 2),
    max_price     DECIMAL(10, 2),
    modal_price   DECIMAL(10, 2) NOT NULL,  -- target variable
    year          SMALLINT,
    month         SMALLINT,
    week          SMALLINT,
    quarter       SMALLINT,
    season        VARCHAR(20),              -- Kharif / Rabi / Zaid
    source        VARCHAR(50) DEFAULT 'data.gov.in',
    created_at    TIMESTAMP DEFAULT NOW(),
    UNIQUE(crop_id, market_id, arrival_date)
);

-- Store model predictions for drift monitoring
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    crop_id         INTEGER REFERENCES crops(id),
    market_id       INTEGER REFERENCES markets(id),
    forecast_date   DATE NOT NULL,
    predicted_price DECIMAL(10, 2),
    model_version   VARCHAR(50),
    confidence_low  DECIMAL(10, 2),
    confidence_high DECIMAL(10, 2),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_price_date     ON price_records(arrival_date);
CREATE INDEX IF NOT EXISTS idx_price_crop     ON price_records(crop_id);
CREATE INDEX IF NOT EXISTS idx_price_market   ON price_records(market_id);
CREATE INDEX IF NOT EXISTS idx_price_crop_mkt ON price_records(crop_id, market_id);
CREATE INDEX IF NOT EXISTS idx_pred_date      ON predictions(forecast_date);
