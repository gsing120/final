import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Tabs, Tab, Table, Badge, Spinner } from 'react-bootstrap';
import { Line, Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend } from 'chart.js';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

const AdvancedVisualization = ({ data, settings }) => {
  const [activeTab, setActiveTab] = useState('price');
  const [timeframe, setTimeframe] = useState('1d');
  const [indicators, setIndicators] = useState([]);
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState(null);
  const [volumeData, setVolumeData] = useState(null);
  const [indicatorData, setIndicatorData] = useState({});
  const [correlationMatrix, setCorrelationMatrix] = useState(null);
  const [heatmapData, setHeatmapData] = useState(null);
  const [patternDetection, setPatternDetection] = useState([]);
  const [supportResistance, setSupportResistance] = useState([]);
  const [volatilityAnalysis, setVolatilityAnalysis] = useState(null);
  const [regressionAnalysis, setRegressionAnalysis] = useState(null);

  // Default settings if not provided
  const defaultSettings = {
    theme: 'light',
    showVolume: true,
    showGrid: true,
    autoRefresh: false,
    refreshInterval: 60000, // 1 minute
    defaultIndicators: ['sma', 'ema', 'rsi'],
    chartType: 'candlestick',
    colorScheme: {
      up: '#26a69a',
      down: '#ef5350',
      volume: 'rgba(100, 100, 100, 0.5)',
      grid: 'rgba(200, 200, 200, 0.2)',
      text: '#333333',
      background: '#ffffff'
    }
  };

  const mergedSettings = { ...defaultSettings, ...settings };

  useEffect(() => {
    if (data) {
      processData();
    }
  }, [data, timeframe, indicators]);

  const processData = () => {
    setLoading(true);
    
    // Simulate processing delay
    setTimeout(() => {
      if (data) {
        // Process price data
        const priceChartData = {
          labels: data.dates,
          datasets: [
            {
              label: 'Price',
              data: data.prices,
              borderColor: mergedSettings.colorScheme.up,
              backgroundColor: 'rgba(38, 166, 154, 0.1)',
              borderWidth: 2,
              pointRadius: 0,
              pointHoverRadius: 5,
              fill: true,
            }
          ]
        };
        setChartData(priceChartData);

        // Process volume data
        const volumeChartData = {
          labels: data.dates,
          datasets: [
            {
              label: 'Volume',
              data: data.volumes,
              backgroundColor: mergedSettings.colorScheme.volume,
              borderWidth: 0,
            }
          ]
        };
        setVolumeData(volumeChartData);

        // Process indicator data
        const processedIndicatorData = {};
        indicators.forEach(indicator => {
          if (data[indicator]) {
            processedIndicatorData[indicator] = {
              labels: data.dates,
              datasets: [
                {
                  label: indicator.toUpperCase(),
                  data: data[indicator],
                  borderColor: getRandomColor(),
                  borderWidth: 2,
                  pointRadius: 0,
                  fill: false,
                }
              ]
            };
          }
        });
        setIndicatorData(processedIndicatorData);

        // Process correlation matrix if available
        if (data.correlationMatrix) {
          setCorrelationMatrix(data.correlationMatrix);
          
          // Create heatmap data
          const heatmapLabels = Object.keys(data.correlationMatrix);
          const heatmapValues = [];
          
          heatmapLabels.forEach(row => {
            const rowData = [];
            heatmapLabels.forEach(col => {
              rowData.push(data.correlationMatrix[row][col]);
            });
            heatmapValues.push(rowData);
          });
          
          setHeatmapData({
            labels: heatmapLabels,
            values: heatmapValues
          });
        }

        // Process pattern detection if available
        if (data.patterns) {
          setPatternDetection(data.patterns);
        }

        // Process support/resistance if available
        if (data.supportResistance) {
          setSupportResistance(data.supportResistance);
        }

        // Process volatility analysis if available
        if (data.volatility) {
          setVolatilityAnalysis(data.volatility);
        }

        // Process regression analysis if available
        if (data.regression) {
          setRegressionAnalysis(data.regression);
        }

        setLoading(false);
      }
    }, 500);
  };

  const getRandomColor = () => {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  };

  const handleAddIndicator = (indicator) => {
    if (!indicators.includes(indicator)) {
      setIndicators([...indicators, indicator]);
    }
  };

  const handleRemoveIndicator = (indicator) => {
    setIndicators(indicators.filter(ind => ind !== indicator));
  };

  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe);
  };

  const renderPriceChart = () => {
    if (!chartData) return <div className="text-center p-5"><Spinner animation="border" /></div>;

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: `Price Chart (${timeframe})`,
        },
        tooltip: {
          mode: 'index',
          intersect: false,
        }
      },
      scales: {
        x: {
          grid: {
            display: mergedSettings.showGrid,
            color: mergedSettings.colorScheme.grid,
          }
        },
        y: {
          grid: {
            display: mergedSettings.showGrid,
            color: mergedSettings.colorScheme.grid,
          }
        }
      },
      interaction: {
        mode: 'nearest',
        axis: 'x',
        intersect: false
      }
    };

    return (
      <div style={{ height: '400px' }}>
        <Line data={chartData} options={options} />
      </div>
    );
  };

  const renderVolumeChart = () => {
    if (!volumeData || !mergedSettings.showVolume) return null;

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        title: {
          display: true,
          text: 'Volume',
        }
      },
      scales: {
        x: {
          display: false,
        },
        y: {
          grid: {
            display: mergedSettings.showGrid,
            color: mergedSettings.colorScheme.grid,
          }
        }
      }
    };

    return (
      <div style={{ height: '150px' }}>
        <Bar data={volumeData} options={options} />
      </div>
    );
  };

  const renderIndicatorCharts = () => {
    if (Object.keys(indicatorData).length === 0) {
      return (
        <div className="text-center p-3">
          <p>No indicators selected</p>
          <Button variant="outline-primary" onClick={() => handleAddIndicator('sma')}>Add SMA</Button>{' '}
          <Button variant="outline-primary" onClick={() => handleAddIndicator('rsi')}>Add RSI</Button>{' '}
          <Button variant="outline-primary" onClick={() => handleAddIndicator('macd')}>Add MACD</Button>
        </div>
      );
    }

    return Object.keys(indicatorData).map(indicator => {
      const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: indicator.toUpperCase(),
          }
        },
        scales: {
          x: {
            display: false,
          },
          y: {
            grid: {
              display: mergedSettings.showGrid,
              color: mergedSettings.colorScheme.grid,
            }
          }
        }
      };

      return (
        <div key={indicator} style={{ height: '150px', marginBottom: '20px' }}>
          <div className="d-flex justify-content-between align-items-center mb-2">
            <h6>{indicator.toUpperCase()}</h6>
            <Button variant="outline-danger" size="sm" onClick={() => handleRemoveIndicator(indicator)}>Remove</Button>
          </div>
          <Line data={indicatorData[indicator]} options={options} />
        </div>
      );
    });
  };

  const renderCorrelationMatrix = () => {
    if (!correlationMatrix) {
      return <div className="text-center p-5">No correlation data available</div>;
    }

    const assets = Object.keys(correlationMatrix);

    return (
      <div className="table-responsive">
        <Table striped bordered hover size="sm">
          <thead>
            <tr>
              <th></th>
              {assets.map(asset => (
                <th key={asset}>{asset}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {assets.map(row => (
              <tr key={row}>
                <td><strong>{row}</strong></td>
                {assets.map(col => {
                  const value = correlationMatrix[row][col];
                  let cellClass = '';
                  if (row !== col) {
                    if (value > 0.7) cellClass = 'bg-danger text-white';
                    else if (value > 0.5) cellClass = 'bg-warning';
                    else if (value < -0.5) cellClass = 'bg-info';
                    else if (value < -0.7) cellClass = 'bg-primary text-white';
                  }
                  return (
                    <td key={`${row}-${col}`} className={cellClass}>
                      {value.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </Table>
      </div>
    );
  };

  const renderHeatmap = () => {
    if (!heatmapData) {
      return <div className="text-center p-5">No heatmap data available</div>;
    }

    // In a real implementation, this would use a proper heatmap visualization library
    return (
      <div className="text-center p-5">
        <p>Heatmap visualization would be rendered here</p>
        <p>Using data with {heatmapData.labels.length} assets</p>
      </div>
    );
  };

  const renderPatternDetection = () => {
    if (!patternDetection || patternDetection.length === 0) {
      return <div className="text-center p-5">No patterns detected</div>;
    }

    return (
      <div className="table-responsive">
        <Table striped bordered hover>
          <thead>
            <tr>
              <th>Pattern</th>
              <th>Start Date</th>
              <th>End Date</th>
              <th>Strength</th>
              <th>Signal</th>
            </tr>
          </thead>
          <tbody>
            {patternDetection.map((pattern, index) => (
              <tr key={index}>
                <td>{pattern.name}</td>
                <td>{pattern.startDate}</td>
                <td>{pattern.endDate}</td>
                <td>
                  <div className="progress">
                    <div 
                      className={`progress-bar ${pattern.strength > 70 ? 'bg-success' : pattern.strength > 40 ? 'bg-warning' : 'bg-danger'}`}
                      role="progressbar" 
                      style={{ width: `${pattern.strength}%` }}
                      aria-valuenow={pattern.strength} 
                      aria-valuemin="0" 
                      aria-valuemax="100"
                    >
                      {pattern.strength}%
                    </div>
                  </div>
                </td>
                <td>
                  <Badge bg={pattern.signal === 'bullish' ? 'success' : 'danger'}>
                    {pattern.signal}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      </div>
    );
  };

  const renderSupportResistance = () => {
    if (!supportResistance || supportResistance.length === 0) {
      return <div className="text-center p-5">No support/resistance levels detected</div>;
    }

    return (
      <div className="table-responsive">
        <Table striped bordered hover>
          <thead>
            <tr>
              <th>Type</th>
              <th>Price Level</th>
              <th>Strength</th>
              <th>Touches</th>
              <th>Age (days)</th>
            </tr>
          </thead>
          <tbody>
            {supportResistance.map((level, index) => (
              <tr key={index}>
                <td>
                  <Badge bg={level.type === 'support' ? 'success' : 'danger'}>
                    {level.type}
                  </Badge>
                </td>
                <td>{level.price.toFixed(2)}</td>
                <td>
                  <div className="progress">
                    <div 
                      className="progress-bar bg-info"
                      role="progressbar" 
                      style={{ width: `${level.strength}%` }}
                      aria-valuenow={level.strength} 
                      aria-valuemin="0" 
                      aria-valuemax="100"
                    >
                      {level.strength}%
                    </div>
                  </div>
                </td>
                <td>{level.touches}</td>
                <td>{level.age}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      </div>
    );
  };

  const renderVolatilityAnalysis = () => {
    if (!volatilityAnalysis) {
      return <div className="text-center p-5">No volatility analysis available</div>;
    }

    const volatilityChartData = {
      labels: volatilityAnalysis.dates,
      datasets: [
        {
          label: 'Historical Volatility',
          data: volatilityAnalysis.historical,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
        },
        {
          label: 'Implied Volatility',
          data: volatilityAnalysis.implied,
          borderColor: 'rgba(153, 102, 255, 1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        }
      ]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Volatility Analysis',
        }
      },
      scales: {
        y: {
          beginAtZero: true
        }
      }
    };

    return (
      <div>
        <div style={{ height: '300px' }}>
          <Line data={volatilityChartData} options={options} />
        </div>
        <div className="row mt-4">
          <div className="col-md-6">
            <Card>
              <Card.Body>
                <Card.Title>Current Volatility</Card.Title>
                <h3>{volatilityAnalysis.current.toFixed(2)}%</h3>
                <p>
                  {volatilityAnalysis.current > volatilityAnalysis.average ? 
                    <span className="text-danger">Above average by {(volatilityAnalysis.current - volatilityAnalysis.average).toFixed(2)}%</span> : 
                    <span className="text-success">Below average by {(volatilityAnalysis.average - volatilityAnalysis.current).toFixed(2)}%</span>
                  }
                </p>
              </Card.Body>
            </Card>
          </div>
          <div className="col-md-6">
            <Card>
              <Card.Body>
                <Card.Title>Volatility Forecast</Card.Title>
                <h3>{volatilityAnalysis.forecast.toFixed(2)}%</h3>
                <p>
                  {volatilityAnalysis.forecast > volatilityAnalysis.current ? 
                    <span className="text-danger">Expected to increase by {(volatilityAnalysis.forecast - volatilityAnalysis.current).toFixed(2)}%</span> : 
                    <span className="text-success">Expected to decrease by {(volatilityAnalysis.current - volatilityAnalysis.forecast).toFixed(2)}%</span>
                  }
                </p>
              </Card.Body>
            </Card>
          </div>
        </div>
      </div>
    );
  };

  const renderRegressionAnalysis = () => {
    if (!regressionAnalysis) {
      return <div className="text-center p-5">No regression analysis available</div>;
    }

    const regressionChartData = {
      labels: regressionAnalysis.dates,
      datasets: [
        {
          label: 'Actual',
          data: regressionAnalysis.actual,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          borderWidth: 2,
          pointRadius: 1,
          fill: false,
        },
        {
          label: 'Predicted',
          data: regressionAnalysis.predicted,
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        },
        {
          label: 'Forecast',
          data: regressionAnalysis.forecast,
          borderColor: 'rgba(255, 159, 64, 1)',
          borderDash: [5, 5],
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        }
      ]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Regression Analysis',
        }
      }
    };

    return (
      <div>
        <div style={{ height: '300px' }}>
          <Line data={regressionChartData} options={options} />
        </div>
        <div className="row mt-4">
          <div className="col-md-4">
            <Card>
              <Card.Body>
                <Card.Title>Model Accuracy</Card.Title>
                <h3>{(regressionAnalysis.r2 * 100).toFixed(2)}%</h3>
                <p>R² value: {regressionAnalysis.r2.toFixed(4)}</p>
              </Card.Body>
            </Card>
          </div>
          <div className="col-md-4">
            <Card>
              <Card.Body>
                <Card.Title>RMSE</Card.Title>
                <h3>{regressionAnalysis.rmse.toFixed(4)}</h3>
                <p>Root Mean Square Error</p>
              </Card.Body>
            </Card>
          </div>
          <div className="col-md-4">
            <Card>
              <Card.Body>
                <Card.Title>Forecast Direction</Card.Title>
                <h3>
                  {regressionAnalysis.forecastDirection === 'up' ? 
                    <span className="text-success">Upward</span> : 
                    <span className="text-danger">Downward</span>
                  }
                </h3>
                <p>Confidence: {regressionAnalysis.forecastConfidence.toFixed(2)}%</p>
              </Card.Body>
            </Card>
          </div>
        </div>
      </div>
    );
  };

  return (
    <Container fluid className="advanced-visualization">
      <Row className="mb-3">
        <Col>
          <div className="d-flex justify-content-between align-items-center">
            <h2>Advanced Visualization</h2>
            <div>
              <Button variant="outline-secondary" size="sm" className="me-2" onClick={() => handleTimeframeChange('1d')}>1D</Button>
              <Button variant="outline-secondary" size="sm" className="me-2" onClick={() => handleTimeframeChange('1w')}>1W</Button>
              <Button variant="outline-secondary" size="sm" className="me-2" onClick={() => handleTimeframeChange('1m')}>1M</Button>
              <Button variant="outline-secondary" size="sm" className="me-2" onClick={() => handleTimeframeChange('3m')}>3M</Button>
              <Button variant="outline-secondary" size="sm" className="me-2" onClick={() => handleTimeframeChange('1y')}>1Y</Button>
              <Button variant="outline-secondary" size="sm" onClick={() => handleTimeframeChange('all')}>All</Button>
            </div>
          </div>
        </Col>
      </Row>

      <Row>
        <Col>
          <Card>
            <Card.Body>
              <Tabs
                activeKey={activeTab}
                onSelect={(k) => setActiveTab(k)}
                className="mb-3"
              >
                <Tab eventKey="price" title="Price Chart">
                  {loading ? (
                    <div className="text-center p-5"><Spinner animation="border" /></div>
                  ) : (
                    <>
                      {renderPriceChart()}
                      {renderVolumeChart()}
                      <div className="mt-4">
                        <h5>Indicators</h5>
                        {renderIndicatorCharts()}
                        <div className="mt-3">
                          <Button variant="outline-primary" size="sm" className="me-2" onClick={() => handleAddIndicator('sma')}>Add SMA</Button>
                          <Button variant="outline-primary" size="sm" className="me-2" onClick={() => handleAddIndicator('ema')}>Add EMA</Button>
                          <Button variant="outline-primary" size="sm" className="me-2" onClick={() => handleAddIndicator('rsi')}>Add RSI</Button>
                          <Button variant="outline-primary" size="sm" className="me-2" onClick={() => handleAddIndicator('macd')}>Add MACD</Button>
                          <Button variant="outline-primary" size="sm" className="me-2" onClick={() => handleAddIndicator('bollinger')}>Add Bollinger</Button>
                          <Button variant="outline-primary" size="sm" onClick={() => handleAddIndicator('atr')}>Add ATR</Button>
                        </div>
                      </div>
                    </>
                  )}
                </Tab>
                <Tab eventKey="correlation" title="Correlation Analysis">
                  {loading ? (
                    <div className="text-center p-5"><Spinner animation="border" /></div>
                  ) : (
                    <>
                      <h5>Correlation Matrix</h5>
                      {renderCorrelationMatrix()}
                      <h5 className="mt-4">Correlation Heatmap</h5>
                      {renderHeatmap()}
                    </>
                  )}
                </Tab>
                <Tab eventKey="patterns" title="Pattern Detection">
                  {loading ? (
                    <div className="text-center p-5"><Spinner animation="border" /></div>
                  ) : (
                    <>
                      <h5>Detected Patterns</h5>
                      {renderPatternDetection()}
                      <h5 className="mt-4">Support & Resistance Levels</h5>
                      {renderSupportResistance()}
                    </>
                  )}
                </Tab>
                <Tab eventKey="volatility" title="Volatility Analysis">
                  {loading ? (
                    <div className="text-center p-5"><Spinner animation="border" /></div>
                  ) : (
                    <>
                      {renderVolatilityAnalysis()}
                    </>
                  )}
                </Tab>
                <Tab eventKey="regression" title="Regression Analysis">
                  {loading ? (
                    <div className="text-center p-5"><Spinner animation="border" /></div>
                  ) : (
                    <>
                      {renderRegressionAnalysis()}
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

export default AdvancedVisualization;
