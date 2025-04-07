import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Table, Badge, Spinner, Alert, Tabs, Tab, Form } from 'react-bootstrap';
import { Line, Bar, Radar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, RadialLinearScale, Title, Tooltip, Legend, Filler } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, RadialLinearScale, Title, Tooltip, Legend, Filler);

const PostMarketInsights = () => {
  const [loading, setLoading] = useState(true);
  const [insights, setInsights] = useState(null);
  const [marketData, setMarketData] = useState(null);
  const [portfolioAnalysis, setPortfolioAnalysis] = useState(null);
  const [tradeRecommendations, setTradeRecommendations] = useState([]);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [activeTab, setActiveTab] = useState('market');
  const [error, setError] = useState(null);
  const [aiExplanation, setAiExplanation] = useState('');
  const [showExplanation, setShowExplanation] = useState(false);

  useEffect(() => {
    fetchInsightsData();
  }, [selectedDate]);

  const fetchInsightsData = async () => {
    setLoading(true);
    setError(null);

    try {
      // In a real implementation, this would be an API call to the backend
      // Simulating API call with setTimeout
      setTimeout(() => {
        // Generate mock data
        generateMockData();
        setLoading(false);
      }, 1500);
    } catch (error) {
      setError('Failed to fetch insights data. Please try again later.');
      setLoading(false);
    }
  };

  const generateMockData = () => {
    // Mock market insights
    const marketInsights = {
      summary: "Markets closed mixed today with technology stocks leading gains while energy stocks lagged. Economic data showed stronger than expected retail sales, suggesting consumer spending remains resilient despite inflation concerns. The Federal Reserve's latest minutes indicated a potential pause in rate hikes if inflation continues to moderate.",
      keyEvents: [
        {
          title: "Fed Minutes Released",
          description: "The Federal Reserve's latest minutes showed a potential pause in rate hikes if inflation continues to moderate.",
          impact: "positive",
          affectedSectors: ["Financials", "Real Estate", "Utilities"]
        },
        {
          title: "Retail Sales Data",
          description: "Retail sales increased 0.7% in March, exceeding economists' expectations of 0.4% growth.",
          impact: "positive",
          affectedSectors: ["Consumer Discretionary", "Consumer Staples"]
        },
        {
          title: "Oil Prices Drop",
          description: "Crude oil prices fell 2.5% on higher than expected inventory data.",
          impact: "negative",
          affectedSectors: ["Energy"]
        }
      ],
      sectorPerformance: [
        { name: "Technology", performance: 1.2 },
        { name: "Healthcare", performance: 0.5 },
        { name: "Financials", performance: 0.3 },
        { name: "Consumer Discretionary", performance: 0.8 },
        { name: "Consumer Staples", performance: -0.2 },
        { name: "Energy", performance: -1.5 },
        { name: "Materials", performance: -0.4 },
        { name: "Industrials", performance: 0.1 },
        { name: "Utilities", performance: 0.4 },
        { name: "Real Estate", performance: 0.6 },
        { name: "Communication Services", performance: 0.9 }
      ],
      marketBreadth: {
        advancers: 285,
        decliners: 215,
        unchanged: 10,
        newHighs: 45,
        newLows: 15,
        advanceVolume: 1.8, // in billions
        declineVolume: 1.2 // in billions
      },
      volatilityAnalysis: {
        vix: 15.8,
        vixChange: -0.5,
        impliedVolatility: {
          current: 16.2,
          historical: 18.5,
          percentile: 35
        },
        putCallRatio: 0.85
      },
      sentimentIndicators: {
        bullishPercent: 55,
        bearishPercent: 25,
        neutralPercent: 20,
        fearGreedIndex: 65, // 0-100, higher means more greed
        shortInterest: {
          overall: -5, // percent change
          notable: [
            { symbol: "XYZ", name: "XYZ Corp", change: 15 },
            { symbol: "ABC", name: "ABC Inc", change: 12 },
            { symbol: "DEF", name: "DEF Co", change: 10 }
          ]
        }
      },
      technicalSignals: {
        spx: {
          macd: "bullish",
          rsi: 58,
          bollingerBands: "middle",
          movingAverages: {
            ma20: "above",
            ma50: "above",
            ma200: "above"
          }
        },
        ndx: {
          macd: "bullish",
          rsi: 62,
          bollingerBands: "upper",
          movingAverages: {
            ma20: "above",
            ma50: "above",
            ma200: "above"
          }
        },
        dji: {
          macd: "neutral",
          rsi: 54,
          bollingerBands: "middle",
          movingAverages: {
            ma20: "above",
            ma50: "above",
            ma200: "above"
          }
        }
      },
      aiMarketRegime: {
        current: "Bullish Trend",
        confidence: 75,
        duration: "15 days",
        characteristics: [
          "Positive momentum across major indices",
          "Broad market participation",
          "Low volatility",
          "Strong earnings reports"
        ]
      }
    };
    
    // Mock portfolio analysis
    const portfolioAnalysis = {
      summary: "Your portfolio outperformed the S&P 500 today by 0.3%, with technology and healthcare holdings contributing most to gains. Risk metrics remain within target ranges, though sector concentration in technology has increased slightly. The AI model suggests maintaining current allocations with potential to increase healthcare exposure on pullbacks.",
      performance: {
        daily: 0.85,
        weekly: 1.75,
        monthly: 3.25,
        ytd: 8.5,
        benchmarkComparison: {
          daily: 0.3, // outperformance vs benchmark
          weekly: 0.5,
          monthly: -0.2,
          ytd: 1.2
        }
      },
      riskAnalysis: {
        volatility: {
          portfolio: 12.5,
          benchmark: 14.2,
          interpretation: "Lower than benchmark"
        },
        drawdown: {
          current: 0,
          maxYTD: 8.5,
          recovery: "Fully recovered"
        },
        sharpeRatio: 1.85,
        sortinoRatio: 2.1,
        beta: 0.92,
        alpha: 2.4,
        rSquared: 0.85
      },
      exposureAnalysis: {
        sectors: [
          { name: "Technology", weight: 35, benchmark: 28, active: 7 },
          { name: "Healthcare", weight: 18, benchmark: 15, active: 3 },
          { name: "Financials", weight: 12, benchmark: 13, active: -1 },
          { name: "Consumer Discretionary", weight: 15, benchmark: 12, active: 3 },
          { name: "Industrials", weight: 8, benchmark: 10, active: -2 },
          { name: "Other", weight: 12, benchmark: 22, active: -10 }
        ],
        factorExposure: {
          value: -0.2,
          growth: 0.5,
          quality: 0.3,
          momentum: 0.4,
          size: -0.1,
          volatility: -0.3
        },
        geographicExposure: {
          domestic: 75,
          international: {
            developed: 20,
            emerging: 5
          }
        }
      },
      topContributors: [
        { symbol: "AAPL", name: "Apple Inc.", contribution: 0.25, performance: 2.1 },
        { symbol: "MSFT", name: "Microsoft Corp.", contribution: 0.18, performance: 1.8 },
        { symbol: "AMZN", name: "Amazon.com Inc.", contribution: 0.15, performance: 2.2 }
      ],
      topDetractors: [
        { symbol: "XOM", name: "Exxon Mobil Corp.", contribution: -0.12, performance: -1.8 },
        { symbol: "JPM", name: "JPMorgan Chase & Co.", contribution: -0.08, performance: -1.2 },
        { symbol: "PG", name: "Procter & Gamble Co.", contribution: -0.05, performance: -0.8 }
      ],
      correlationMatrix: [
        { name: "Portfolio", portfolio: 1.0, sp500: 0.85, nasdaq: 0.88, russell2000: 0.75, bonds: -0.2 },
        { name: "S&P 500", portfolio: 0.85, sp500: 1.0, nasdaq: 0.92, russell2000: 0.8, bonds: -0.25 },
        { name: "Nasdaq", portfolio: 0.88, sp500: 0.92, nasdaq: 1.0, russell2000: 0.78, bonds: -0.3 },
        { name: "Russell 2000", portfolio: 0.75, sp500: 0.8, nasdaq: 0.78, russell2000: 1.0, bonds: -0.15 },
        { name: "Bonds", portfolio: -0.2, sp500: -0.25, nasdaq: -0.3, russell2000: -0.15, bonds: 1.0 }
      ],
      aiInsights: [
        "Portfolio shows strong momentum alignment with current market regime",
        "Technology exposure has been beneficial but consider trimming on further strength",
        "Healthcare positions are well-positioned for current economic conditions",
        "Consider increasing exposure to quality factors for defensive positioning",
        "International diversification remains below target and could be increased"
      ]
    };
    
    // Mock trade recommendations
    const tradeRecommendations = [
      {
        type: "entry",
        symbol: "NVDA",
        name: "NVIDIA Corporation",
        strategy: "Momentum Breakout",
        direction: "long",
        price: 925.50,
        targetPrice: 980.00,
        stopLoss: 890.00,
        timeframe: "1-2 weeks",
        confidence: 85,
        reasoning: "Breaking out of consolidation pattern with strong volume. AI demand continues to drive growth. Recent product announcements well-received by market.",
        technicalFactors: [
          "Price above all major moving averages",
          "RSI at 65, showing strength without being overbought",
          "Volume increasing on breakout",
          "Bullish MACD crossover"
        ],
        fundamentalFactors: [
          "Strong earnings growth",
          "Expanding market share in AI chips",
          "Positive analyst revisions",
          "Industry leadership position"
        ],
        riskFactors: [
          "Valuation at premium to historical average",
          "Semiconductor sector cyclicality",
          "Potential regulatory concerns"
        ]
      },
      {
        type: "exit",
        symbol: "XOM",
        name: "Exxon Mobil Corporation",
        strategy: "Trend Following",
        direction: "long",
        price: 118.25,
        originalEntry: 105.50,
        profitLoss: 12.1,
        timeframe: "immediate",
        confidence: 75,
        reasoning: "Breaking below key support levels with increasing volume. Energy sector facing headwinds from inventory data and demand concerns.",
        technicalFactors: [
          "Price broke below 50-day moving average",
          "RSI showing negative divergence",
          "Volume increasing on down days",
          "Head and shoulders pattern forming"
        ],
        fundamentalFactors: [
          "Oil inventory data higher than expected",
          "Demand forecasts being revised lower",
          "Sector rotation out of energy",
          "Valuation still reasonable but momentum shifting"
        ],
        riskFactors: [
          "Potential OPEC+ production cuts could support prices",
          "Geopolitical tensions could spike energy prices",
          "Dividend yield provides support"
        ]
      },
      {
        type: "adjustment",
        symbol: "MSFT",
        name: "Microsoft Corporation",
        strategy: "Position Sizing",
        action: "increase allocation",
        currentAllocation: 3.5,
        targetAllocation: 5.0,
        price: 415.75,
        timeframe: "gradual entry",
        confidence: 80,
        reasoning: "Strong fundamental performance and technical strength warrant increased position size. AI initiatives and cloud growth continue to drive results.",
        technicalFactors: [
          "Steady uptrend with minimal volatility",
          "Price consolidating near all-time highs",
          "Relative strength versus sector and market",
          "Healthy pullbacks finding support at moving averages"
        ],
        fundamentalFactors: [
          "Cloud revenue growth exceeding expectations",
          "AI integration across product suite",
          "Strong free cash flow generation",
          "Consistent margin expansion"
        ],
        riskFactors: [
          "Valuation above historical average",
          "Regulatory scrutiny of big tech",
          "Competitive pressures in cloud space"
        ]
      }
    ];
    
    // AI explanation for market analysis
    const aiExplanation = `
    # Market Analysis Methodology

    ## Data Sources
    The market analysis integrates data from multiple sources:
    - Real-time and historical price data across major indices, sectors, and individual securities
    - Economic indicators including GDP, inflation, employment, and retail sales
    - Central bank communications and policy decisions
    - Corporate earnings reports and guidance
    - Options market data for implied volatility and sentiment
    - Technical indicators across multiple timeframes
    - Alternative data sources including social media sentiment and news flow

    ## Analytical Framework
    The analysis employs a multi-layered approach:

    ### 1. Market Regime Identification
    Using unsupervised learning algorithms to classify the current market environment into one of several regimes:
    - Bullish Trend
    - Bearish Trend
    - Range-Bound
    - High Volatility
    - Low Volatility Mean-Reversion
    - Sector Rotation

    The classification considers volatility patterns, breadth metrics, correlation structures, and momentum factors with a 75% confidence threshold for regime assignment.

    ### 2. Factor Analysis
    Decomposing market movements into key factors:
    - Value/Growth dynamics
    - Quality metrics
    - Momentum signals
    - Size effects
    - Volatility characteristics
    - Liquidity conditions

    ### 3. Intermarket Analysis
    Examining relationships between:
    - Equity markets (domestic, international, emerging)
    - Fixed income (yields, spreads, curve shape)
    - Currencies
    - Commodities
    - Credit markets

    ### 4. Sentiment Evaluation
    Quantifying market sentiment through:
    - Options market positioning (put/call ratios, skew)
    - Investor surveys
    - Fund flows
    - Short interest
    - Social media and news sentiment analysis

    ## Predictive Modeling
    The system employs ensemble methods combining:
    - Statistical time series models
    - Machine learning classifiers
    - Deep learning networks trained on historical market regimes
    - Natural language processing for news and earnings call analysis

    Models are continuously evaluated and recalibrated based on performance metrics, with uncertainty quantification provided for all predictions.

    ## Interpretation Framework
    The final analysis integrates quantitative signals with qualitative context, considering:
    - Historical analogues to current conditions
    - Structural market changes
    - Policy environment
    - Liquidity conditions
    - Positioning extremes

    Confidence levels are assigned based on signal strength, consistency across timeframes, and historical accuracy in similar market conditions.
    `;

    setMarketData(marketInsights);
    setPortfolioAnalysis(portfolioAnalysis);
    setTradeRecommendations(tradeRecommendations);
    setAiExplanation(aiExplanation);
    
    // Combine all insights
    setInsights({
      market: marketInsights,
      portfolio: portfolioAnalysis,
      trades: tradeRecommendations
    });
  };

  const handleDateChange = (e) => {
    setSelectedDate(e.target.value);
  };

  const handleRefreshData = () => {
    fetchInsightsData();
  };

  const renderSectorPerformance = () => {
    if (!marketData || !marketData.sectorPerformance) return null;

    const chartData = {
      labels: marketData.sectorPerformance.map(sector => sector.name),
      datasets: [
        {
          label: 'Performance (%)',
          data: marketData.sectorPerformance.map(sector => sector.performance),
          backgroundColor: marketData.sectorPerformance.map(sector => 
            sector.performance >= 0 ? 'rgba(75, 192, 192, 0.7)' : 'rgba(255, 99, 132, 0.7)'
          ),
          borderColor: marketData.sectorPerformance.map(sector => 
            sector.performance >= 0 ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)'
          ),
          borderWidth: 1,
        }
      ]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        }
      },
      scales: {
        y: {
          grid: {
            color: 'rgba(200, 200, 200, 0.2)',
          }
        }
      }
    };

    return (
      <div style={{ height: '300px' }}>
        <Bar data={chartData} options={options} />
      </div>
    );
  };

  const renderFactorExposure = () => {
    if (!portfolioAnalysis || !portfolioAnalysis.exposureAnalysis || !portfolioAnalysis.exposureAnalysis.factorExposure) return null;

    const factors = portfolioAnalysis.exposureAnalysis.factorExposure;
    
    const chartData = {
      labels: Object.keys(factors).map(factor => factor.charAt(0).toUpperCase() + factor.slice(1)),
      datasets: [
        {
          label: 'Factor Exposure',
          data: Object.values(factors),
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1,
          pointBackgroundColor: 'rgba(54, 162, 235, 1)',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: 'rgba(54, 162, 235, 1)',
          fill: true
        }
      ]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: {
            display: true
          },
          suggestedMin: -1,
          suggestedMax: 1
        }
      }
    };

    return (
      <div style={{ height: '300px' }}>
        <Radar data={chartData} options={options} />
      </div>
    );
  };

  const renderSectorAllocation = () => {
    if (!portfolioAnalysis || !portfolioAnalysis.exposureAnalysis || !portfolioAnalysis.exposureAnalysis.sectors) return null;

    const chartData = {
      labels: portfolioAnalysis.exposureAnalysis.sectors.map(sector => sector.name),
      datasets: [
        {
          label: 'Portfolio Weight (%)',
          data: portfolioAnalysis.exposureAnalysis.sectors.map(sector => sector.weight),
          backgroundColor: 'rgba(75, 192, 192, 0.7)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1,
        },
        {
          label: 'Benchmark Weight (%)',
          data: portfolioAnalysis.exposureAnalysis.sectors.map(sector => sector.benchmark),
          backgroundColor: 'rgba(153, 102, 255, 0.7)',
          borderColor: 'rgba(153, 102, 255, 1)',
          borderWidth: 1,
        }
      ]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        }
      },
      scales: {
        y: {
          grid: {
            color: 'rgba(200, 200, 200, 0.2)',
          }
        }
      }
    };

    return (
      <div style={{ height: '300px' }}>
        <Bar data={chartData} options={options} />
      </div>
    );
  };

  return (
    <Container fluid className="post-market-insights">
      <Row className="mb-3">
        <Col>
          <div className="d-flex justify-content-between align-items-center">
            <h2>Post-Market AI Insights</h2>
            <div className="d-flex align-items-center">
              <Form.Control
                type="date"
                value={selectedDate}
                onChange={handleDateChange}
                style={{ width: '200px' }}
                className="me-2"
              />
              <Button 
                variant="outline-primary" 
                size="sm" 
                onClick={handleRefreshData}
                disabled={loading}
              >
                {loading ? <Spinner animation="border" size="sm" /> : 'Refresh'}
              </Button>
            </div>
          </div>
        </Col>
      </Row>

      {error && (
        <Row className="mb-3">
          <Col>
            <Alert variant="danger">{error}</Alert>
          </Col>
        </Row>
      )}

      <Row className="mb-4">
        <Col>
          <Card>
            <Card.Body>
              <Tabs
                activeKey={activeTab}
                onSelect={(k) => setActiveTab(k)}
                className="mb-3"
              >
                <Tab eventKey="market" title="Market Analysis">
                  {loading ? (
                    <div className="text-center p-5"><Spinner animation="border" /></div>
                  ) : (
                    <>
                      <Row className="mb-4">
                        <Col lg={8}>
                          <h4>Market Summary</h4>
                          <p>{marketData?.summary}</p>
                          
                          <h5 className="mt-4">Key Events</h5>
                          {marketData?.keyEvents.map((event, index) => (
                            <Card key={index} className="mb-3">
                              <Card.Body>
                                <div className="d-flex justify-content-between align-items-center">
                                  <h6 className="mb-0">{event.title}</h6>
                                  <Badge bg={event.impact === 'positive' ? 'success' : event.impact === 'negative' ? 'danger' : 'warning'}>
                                    {event.impact.toUpperCase()}
                                  </Badge>
                                </div>
                                <p className="mb-1 mt-2">{event.description}</p>
                                <small className="text-muted">
                                  Affected sectors: {event.affectedSectors.join(', ')}
                                </small>
                              </Card.Body>
                            </Card>
                          ))}
                          
                          <h5 className="mt-4">AI Market Regime Analysis</h5>
                          <Card>
                            <Card.Body>
                              <div className="d-flex justify-content-between align-items-center mb-3">
                                <div>
                                  <h6 className="mb-0">Current Regime: {marketData?.aiMarketRegime.current}</h6>
                                  <small className="text-muted">Duration: {marketData?.aiMarketRegime.duration}</small>
                                </div>
                                <Badge bg="info">
                                  Confidence: {marketData?.aiMarketRegime.confidence}%
                                </Badge>
                              </div>
                              <h6>Characteristics:</h6>
                              <ul>
                                {marketData?.aiMarketRegime.characteristics.map((item, index) => (
                                  <li key={index}>{item}</li>
                                ))}
                              </ul>
                              <div className="text-end">
                                <Button 
                                  variant="link" 
                                  size="sm"
                                  onClick={() => setShowExplanation(!showExplanation)}
                                >
                                  {showExplanation ? 'Hide Methodology' : 'View AI Methodology'}
                                </Button>
                              </div>
                            </Card.Body>
                          </Card>
                          
                          {showExplanation && (
                            <Card className="mt-3">
                              <Card.Body>
                                <h5>AI Analysis Methodology</h5>
                                <div style={{ whiteSpace: 'pre-line' }}>
                                  {aiExplanation}
                                </div>
                              </Card.Body>
                            </Card>
                          )}
                        </Col>
                        <Col lg={4}>
                          <Card className="mb-3">
                            <Card.Body>
                              <Card.Title>Sector Performance</Card.Title>
                              {renderSectorPerformance()}
                            </Card.Body>
                          </Card>
                          
                          <Card className="mb-3">
                            <Card.Body>
                              <Card.Title>Market Breadth</Card.Title>
                              <Table bordered size="sm">
                                <tbody>
                                  <tr>
                                    <td>Advancers</td>
                                    <td>{marketData?.marketBreadth.advancers}</td>
                                  </tr>
                                  <tr>
                                    <td>Decliners</td>
                                    <td>{marketData?.marketBreadth.decliners}</td>
                                  </tr>
                                  <tr>
                                    <td>Unchanged</td>
                                    <td>{marketData?.marketBreadth.unchanged}</td>
                                  </tr>
                                  <tr>
                                    <td>New Highs</td>
                                    <td>{marketData?.marketBreadth.newHighs}</td>
                                  </tr>
                                  <tr>
                                    <td>New Lows</td>
                                    <td>{marketData?.marketBreadth.newLows}</td>
                                  </tr>
                                  <tr>
                                    <td>Advance Volume</td>
                                    <td>{marketData?.marketBreadth.advanceVolume}B</td>
                                  </tr>
                                  <tr>
                                    <td>Decline Volume</td>
                                    <td>{marketData?.marketBreadth.declineVolume}B</td>
                                  </tr>
                                </tbody>
                              </Table>
                            </Card.Body>
                          </Card>
                          
                          <Card>
                            <Card.Body>
                              <Card.Title>Volatility & Sentiment</Card.Title>
                              <Table bordered size="sm">
                                <tbody>
                                  <tr>
                                    <td>VIX</td>
                                    <td>
                                      {marketData?.volatilityAnalysis.vix}
                                      <span className={marketData?.volatilityAnalysis.vixChange >= 0 ? 'text-danger ms-2' : 'text-success ms-2'}>
                                        {marketData?.volatilityAnalysis.vixChange >= 0 ? '+' : ''}{marketData?.volatilityAnalysis.vixChange}
                                      </span>
                                    </td>
                                  </tr>
                                  <tr>
                                    <td>Put/Call Ratio</td>
                                    <td>{marketData?.volatilityAnalysis.putCallRatio}</td>
                                  </tr>
                                  <tr>
                                    <td>Implied Volatility Percentile</td>
                                    <td>{marketData?.volatilityAnalysis.impliedVolatility.percentile}%</td>
                                  </tr>
                                  <tr>
                                    <td>Bullish Sentiment</td>
                                    <td>{marketData?.sentimentIndicators.bullishPercent}%</td>
                                  </tr>
                                  <tr>
                                    <td>Bearish Sentiment</td>
                                    <td>{marketData?.sentimentIndicators.bearishPercent}%</td>
                                  </tr>
                                  <tr>
                                    <td>Fear/Greed Index</td>
                                    <td>
                                      {marketData?.sentimentIndicators.fearGreedIndex}
                                      <span className="ms-2">
                                        ({marketData?.sentimentIndicators.fearGreedIndex > 75 ? 'Extreme Greed' : 
                                          marketData?.sentimentIndicators.fearGreedIndex > 60 ? 'Greed' :
                                          marketData?.sentimentIndicators.fearGreedIndex > 40 ? 'Neutral' :
                                          marketData?.sentimentIndicators.fearGreedIndex > 25 ? 'Fear' : 'Extreme Fear'})
                                      </span>
                                    </td>
                                  </tr>
                                </tbody>
                              </Table>
                            </Card.Body>
                          </Card>
                        </Col>
                      </Row>
                      
                      <Row>
                        <Col>
                          <h5>Technical Signals</h5>
                          <Table bordered responsive>
                            <thead>
                              <tr>
                                <th>Index</th>
                                <th>MACD</th>
                                <th>RSI</th>
                                <th>Bollinger Bands</th>
                                <th>20-Day MA</th>
                                <th>50-Day MA</th>
                                <th>200-Day MA</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr>
                                <td>S&P 500</td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.spx.macd === 'bullish' ? 'success' :
                                    marketData?.technicalSignals.spx.macd === 'bearish' ? 'danger' : 'warning'
                                  }>
                                    {marketData?.technicalSignals.spx.macd.toUpperCase()}
                                  </Badge>
                                </td>
                                <td>
                                  {marketData?.technicalSignals.spx.rsi}
                                  <span className="ms-1">
                                    ({marketData?.technicalSignals.spx.rsi > 70 ? 'Overbought' : 
                                      marketData?.technicalSignals.spx.rsi < 30 ? 'Oversold' : 'Neutral'})
                                  </span>
                                </td>
                                <td>{marketData?.technicalSignals.spx.bollingerBands.toUpperCase()}</td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.spx.movingAverages.ma20 === 'above' ? 'success' : 'danger'
                                  }>
                                    {marketData?.technicalSignals.spx.movingAverages.ma20.toUpperCase()}
                                  </Badge>
                                </td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.spx.movingAverages.ma50 === 'above' ? 'success' : 'danger'
                                  }>
                                    {marketData?.technicalSignals.spx.movingAverages.ma50.toUpperCase()}
                                  </Badge>
                                </td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.spx.movingAverages.ma200 === 'above' ? 'success' : 'danger'
                                  }>
                                    {marketData?.technicalSignals.spx.movingAverages.ma200.toUpperCase()}
                                  </Badge>
                                </td>
                              </tr>
                              <tr>
                                <td>Nasdaq</td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.ndx.macd === 'bullish' ? 'success' :
                                    marketData?.technicalSignals.ndx.macd === 'bearish' ? 'danger' : 'warning'
                                  }>
                                    {marketData?.technicalSignals.ndx.macd.toUpperCase()}
                                  </Badge>
                                </td>
                                <td>
                                  {marketData?.technicalSignals.ndx.rsi}
                                  <span className="ms-1">
                                    ({marketData?.technicalSignals.ndx.rsi > 70 ? 'Overbought' : 
                                      marketData?.technicalSignals.ndx.rsi < 30 ? 'Oversold' : 'Neutral'})
                                  </span>
                                </td>
                                <td>{marketData?.technicalSignals.ndx.bollingerBands.toUpperCase()}</td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.ndx.movingAverages.ma20 === 'above' ? 'success' : 'danger'
                                  }>
                                    {marketData?.technicalSignals.ndx.movingAverages.ma20.toUpperCase()}
                                  </Badge>
                                </td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.ndx.movingAverages.ma50 === 'above' ? 'success' : 'danger'
                                  }>
                                    {marketData?.technicalSignals.ndx.movingAverages.ma50.toUpperCase()}
                                  </Badge>
                                </td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.ndx.movingAverages.ma200 === 'above' ? 'success' : 'danger'
                                  }>
                                    {marketData?.technicalSignals.ndx.movingAverages.ma200.toUpperCase()}
                                  </Badge>
                                </td>
                              </tr>
                              <tr>
                                <td>Dow Jones</td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.dji.macd === 'bullish' ? 'success' :
                                    marketData?.technicalSignals.dji.macd === 'bearish' ? 'danger' : 'warning'
                                  }>
                                    {marketData?.technicalSignals.dji.macd.toUpperCase()}
                                  </Badge>
                                </td>
                                <td>
                                  {marketData?.technicalSignals.dji.rsi}
                                  <span className="ms-1">
                                    ({marketData?.technicalSignals.dji.rsi > 70 ? 'Overbought' : 
                                      marketData?.technicalSignals.dji.rsi < 30 ? 'Oversold' : 'Neutral'})
                                  </span>
                                </td>
                                <td>{marketData?.technicalSignals.dji.bollingerBands.toUpperCase()}</td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.dji.movingAverages.ma20 === 'above' ? 'success' : 'danger'
                                  }>
                                    {marketData?.technicalSignals.dji.movingAverages.ma20.toUpperCase()}
                                  </Badge>
                                </td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.dji.movingAverages.ma50 === 'above' ? 'success' : 'danger'
                                  }>
                                    {marketData?.technicalSignals.dji.movingAverages.ma50.toUpperCase()}
                                  </Badge>
                                </td>
                                <td>
                                  <Badge bg={
                                    marketData?.technicalSignals.dji.movingAverages.ma200 === 'above' ? 'success' : 'danger'
                                  }>
                                    {marketData?.technicalSignals.dji.movingAverages.ma200.toUpperCase()}
                                  </Badge>
                                </td>
                              </tr>
                            </tbody>
                          </Table>
                        </Col>
                      </Row>
                    </>
                  )}
                </Tab>
                
                <Tab eventKey="portfolio" title="Portfolio Analysis">
                  {loading ? (
                    <div className="text-center p-5"><Spinner animation="border" /></div>
                  ) : (
                    <>
                      <Row className="mb-4">
                        <Col lg={8}>
                          <h4>Portfolio Summary</h4>
                          <p>{portfolioAnalysis?.summary}</p>
                          
                          <h5 className="mt-4">Performance</h5>
                          <Table bordered>
                            <thead>
                              <tr>
                                <th>Period</th>
                                <th>Portfolio Return</th>
                                <th>vs. Benchmark</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr>
                                <td>Daily</td>
                                <td className={portfolioAnalysis?.performance.daily >= 0 ? 'text-success' : 'text-danger'}>
                                  {portfolioAnalysis?.performance.daily >= 0 ? '+' : ''}{portfolioAnalysis?.performance.daily}%
                                </td>
                                <td className={portfolioAnalysis?.performance.benchmarkComparison.daily >= 0 ? 'text-success' : 'text-danger'}>
                                  {portfolioAnalysis?.performance.benchmarkComparison.daily >= 0 ? '+' : ''}{portfolioAnalysis?.performance.benchmarkComparison.daily}%
                                </td>
                              </tr>
                              <tr>
                                <td>Weekly</td>
                                <td className={portfolioAnalysis?.performance.weekly >= 0 ? 'text-success' : 'text-danger'}>
                                  {portfolioAnalysis?.performance.weekly >= 0 ? '+' : ''}{portfolioAnalysis?.performance.weekly}%
                                </td>
                                <td className={portfolioAnalysis?.performance.benchmarkComparison.weekly >= 0 ? 'text-success' : 'text-danger'}>
                                  {portfolioAnalysis?.performance.benchmarkComparison.weekly >= 0 ? '+' : ''}{portfolioAnalysis?.performance.benchmarkComparison.weekly}%
                                </td>
                              </tr>
                              <tr>
                                <td>Monthly</td>
                                <td className={portfolioAnalysis?.performance.monthly >= 0 ? 'text-success' : 'text-danger'}>
                                  {portfolioAnalysis?.performance.monthly >= 0 ? '+' : ''}{portfolioAnalysis?.performance.monthly}%
                                </td>
                                <td className={portfolioAnalysis?.performance.benchmarkComparison.monthly >= 0 ? 'text-success' : 'text-danger'}>
                                  {portfolioAnalysis?.performance.benchmarkComparison.monthly >= 0 ? '+' : ''}{portfolioAnalysis?.performance.benchmarkComparison.monthly}%
                                </td>
                              </tr>
                              <tr>
                                <td>Year-to-Date</td>
                                <td className={portfolioAnalysis?.performance.ytd >= 0 ? 'text-success' : 'text-danger'}>
                                  {portfolioAnalysis?.performance.ytd >= 0 ? '+' : ''}{portfolioAnalysis?.performance.ytd}%
                                </td>
                                <td className={portfolioAnalysis?.performance.benchmarkComparison.ytd >= 0 ? 'text-success' : 'text-danger'}>
                                  {portfolioAnalysis?.performance.benchmarkComparison.ytd >= 0 ? '+' : ''}{portfolioAnalysis?.performance.benchmarkComparison.ytd}%
                                </td>
                              </tr>
                            </tbody>
                          </Table>
                          
                          <Row className="mt-4">
                            <Col md={6}>
                              <h5>Top Contributors</h5>
                              <Table bordered size="sm">
                                <thead>
                                  <tr>
                                    <th>Symbol</th>
                                    <th>Name</th>
                                    <th>Contribution</th>
                                    <th>Performance</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {portfolioAnalysis?.topContributors.map((item, index) => (
                                    <tr key={index}>
                                      <td>{item.symbol}</td>
                                      <td>{item.name}</td>
                                      <td className="text-success">+{item.contribution}%</td>
                                      <td className="text-success">+{item.performance}%</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </Table>
                            </Col>
                            <Col md={6}>
                              <h5>Top Detractors</h5>
                              <Table bordered size="sm">
                                <thead>
                                  <tr>
                                    <th>Symbol</th>
                                    <th>Name</th>
                                    <th>Contribution</th>
                                    <th>Performance</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {portfolioAnalysis?.topDetractors.map((item, index) => (
                                    <tr key={index}>
                                      <td>{item.symbol}</td>
                                      <td>{item.name}</td>
                                      <td className="text-danger">{item.contribution}%</td>
                                      <td className="text-danger">{item.performance}%</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </Table>
                            </Col>
                          </Row>
                          
                          <h5 className="mt-4">AI Insights</h5>
                          <Card>
                            <Card.Body>
                              <ul>
                                {portfolioAnalysis?.aiInsights.map((insight, index) => (
                                  <li key={index}>{insight}</li>
                                ))}
                              </ul>
                            </Card.Body>
                          </Card>
                        </Col>
                        <Col lg={4}>
                          <Card className="mb-3">
                            <Card.Body>
                              <Card.Title>Sector Allocation</Card.Title>
                              {renderSectorAllocation()}
                            </Card.Body>
                          </Card>
                          
                          <Card className="mb-3">
                            <Card.Body>
                              <Card.Title>Factor Exposure</Card.Title>
                              {renderFactorExposure()}
                            </Card.Body>
                          </Card>
                          
                          <Card>
                            <Card.Body>
                              <Card.Title>Risk Metrics</Card.Title>
                              <Table bordered size="sm">
                                <tbody>
                                  <tr>
                                    <td>Volatility (Portfolio)</td>
                                    <td>{portfolioAnalysis?.riskAnalysis.volatility.portfolio}%</td>
                                  </tr>
                                  <tr>
                                    <td>Volatility (Benchmark)</td>
                                    <td>{portfolioAnalysis?.riskAnalysis.volatility.benchmark}%</td>
                                  </tr>
                                  <tr>
                                    <td>Max Drawdown YTD</td>
                                    <td className="text-danger">{portfolioAnalysis?.riskAnalysis.drawdown.maxYTD}%</td>
                                  </tr>
                                  <tr>
                                    <td>Sharpe Ratio</td>
                                    <td>{portfolioAnalysis?.riskAnalysis.sharpeRatio}</td>
                                  </tr>
                                  <tr>
                                    <td>Sortino Ratio</td>
                                    <td>{portfolioAnalysis?.riskAnalysis.sortinoRatio}</td>
                                  </tr>
                                  <tr>
                                    <td>Beta</td>
                                    <td>{portfolioAnalysis?.riskAnalysis.beta}</td>
                                  </tr>
                                  <tr>
                                    <td>Alpha</td>
                                    <td className={portfolioAnalysis?.riskAnalysis.alpha >= 0 ? 'text-success' : 'text-danger'}>
                                      {portfolioAnalysis?.riskAnalysis.alpha >= 0 ? '+' : ''}{portfolioAnalysis?.riskAnalysis.alpha}%
                                    </td>
                                  </tr>
                                  <tr>
                                    <td>R-Squared</td>
                                    <td>{portfolioAnalysis?.riskAnalysis.rSquared}</td>
                                  </tr>
                                </tbody>
                              </Table>
                            </Card.Body>
                          </Card>
                        </Col>
                      </Row>
                      
                      <Row>
                        <Col>
                          <h5>Correlation Matrix</h5>
                          <div className="table-responsive">
                            <Table bordered size="sm">
                              <thead>
                                <tr>
                                  <th></th>
                                  <th>Portfolio</th>
                                  <th>S&P 500</th>
                                  <th>Nasdaq</th>
                                  <th>Russell 2000</th>
                                  <th>Bonds</th>
                                </tr>
                              </thead>
                              <tbody>
                                {portfolioAnalysis?.correlationMatrix.map((row, index) => (
                                  <tr key={index}>
                                    <td><strong>{row.name}</strong></td>
                                    <td>{row.portfolio.toFixed(2)}</td>
                                    <td>{row.sp500.toFixed(2)}</td>
                                    <td>{row.nasdaq.toFixed(2)}</td>
                                    <td>{row.russell2000.toFixed(2)}</td>
                                    <td>{row.bonds.toFixed(2)}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </Table>
                          </div>
                        </Col>
                      </Row>
                    </>
                  )}
                </Tab>
                
                <Tab eventKey="trades" title="Trade Recommendations">
                  {loading ? (
                    <div className="text-center p-5"><Spinner animation="border" /></div>
                  ) : (
                    <>
                      {tradeRecommendations.map((trade, index) => (
                        <Card key={index} className="mb-4">
                          <Card.Header className="d-flex justify-content-between align-items-center">
                            <div>
                              <Badge bg={
                                trade.type === 'entry' ? 'success' : 
                                trade.type === 'exit' ? 'danger' : 'info'
                              } className="me-2">
                                {trade.type.toUpperCase()}
                              </Badge>
                              <strong>{trade.symbol}</strong> - {trade.name}
                            </div>
                            <Badge bg={
                              trade.confidence >= 80 ? 'success' :
                              trade.confidence >= 60 ? 'warning' : 'secondary'
                            }>
                              Confidence: {trade.confidence}%
                            </Badge>
                          </Card.Header>
                          <Card.Body>
                            <Row>
                              <Col md={8}>
                                <h5>Reasoning</h5>
                                <p>{trade.reasoning}</p>
                                
                                <Row className="mt-3">
                                  <Col md={6}>
                                    <h6>Technical Factors</h6>
                                    <ul>
                                      {trade.technicalFactors.map((factor, idx) => (
                                        <li key={idx}>{factor}</li>
                                      ))}
                                    </ul>
                                  </Col>
                                  <Col md={6}>
                                    <h6>Fundamental Factors</h6>
                                    <ul>
                                      {trade.fundamentalFactors.map((factor, idx) => (
                                        <li key={idx}>{factor}</li>
                                      ))}
                                    </ul>
                                  </Col>
                                </Row>
                                
                                <h6 className="mt-3">Risk Factors</h6>
                                <ul>
                                  {trade.riskFactors.map((factor, idx) => (
                                    <li key={idx}>{factor}</li>
                                  ))}
                                </ul>
                              </Col>
                              <Col md={4}>
                                <Card>
                                  <Card.Body>
                                    <h6>Trade Details</h6>
                                    <Table bordered size="sm">
                                      <tbody>
                                        <tr>
                                          <td>Strategy</td>
                                          <td>{trade.strategy}</td>
                                        </tr>
                                        {trade.type === 'entry' && (
                                          <>
                                            <tr>
                                              <td>Direction</td>
                                              <td>
                                                <Badge bg={trade.direction === 'long' ? 'success' : 'danger'}>
                                                  {trade.direction.toUpperCase()}
                                                </Badge>
                                              </td>
                                            </tr>
                                            <tr>
                                              <td>Entry Price</td>
                                              <td>${trade.price.toFixed(2)}</td>
                                            </tr>
                                            <tr>
                                              <td>Target Price</td>
                                              <td className="text-success">${trade.targetPrice.toFixed(2)}</td>
                                            </tr>
                                            <tr>
                                              <td>Stop Loss</td>
                                              <td className="text-danger">${trade.stopLoss.toFixed(2)}</td>
                                            </tr>
                                            <tr>
                                              <td>Risk/Reward</td>
                                              <td>
                                                {((trade.targetPrice - trade.price) / (trade.price - trade.stopLoss)).toFixed(2)}
                                              </td>
                                            </tr>
                                          </>
                                        )}
                                        {trade.type === 'exit' && (
                                          <>
                                            <tr>
                                              <td>Direction</td>
                                              <td>
                                                <Badge bg={trade.direction === 'long' ? 'success' : 'danger'}>
                                                  {trade.direction.toUpperCase()}
                                                </Badge>
                                              </td>
                                            </tr>
                                            <tr>
                                              <td>Current Price</td>
                                              <td>${trade.price.toFixed(2)}</td>
                                            </tr>
                                            <tr>
                                              <td>Entry Price</td>
                                              <td>${trade.originalEntry.toFixed(2)}</td>
                                            </tr>
                                            <tr>
                                              <td>P&L</td>
                                              <td className={trade.profitLoss >= 0 ? 'text-success' : 'text-danger'}>
                                                {trade.profitLoss >= 0 ? '+' : ''}{trade.profitLoss.toFixed(2)}%
                                              </td>
                                            </tr>
                                          </>
                                        )}
                                        {trade.type === 'adjustment' && (
                                          <>
                                            <tr>
                                              <td>Action</td>
                                              <td>{trade.action}</td>
                                            </tr>
                                            <tr>
                                              <td>Current Allocation</td>
                                              <td>{trade.currentAllocation}%</td>
                                            </tr>
                                            <tr>
                                              <td>Target Allocation</td>
                                              <td>{trade.targetAllocation}%</td>
                                            </tr>
                                            <tr>
                                              <td>Current Price</td>
                                              <td>${trade.price.toFixed(2)}</td>
                                            </tr>
                                          </>
                                        )}
                                        <tr>
                                          <td>Timeframe</td>
                                          <td>{trade.timeframe}</td>
                                        </tr>
                                      </tbody>
                                    </Table>
                                    <div className="d-grid gap-2 mt-3">
                                      <Button variant="primary" size="sm">
                                        Execute Trade
                                      </Button>
                                      <Button variant="outline-secondary" size="sm">
                                        Save for Later
                                      </Button>
                                    </div>
                                  </Card.Body>
                                </Card>
                              </Col>
                            </Row>
                          </Card.Body>
                        </Card>
                      ))}
                    </>
                  )}
                </Tab>
              </Tabs>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default PostMarketInsights;
