import React, { useState, useEffect } from 'react';
import { Upload, Settings, ChevronDown, MessageSquare, Send, User, Bot, BarChart3, TrendingUp, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
// The 'recharts' import has been removed as it is no longer needed.


const COLUMNS_DICT = {
  // ... (Your existing large COLUMNS_DICT object remains here, no changes needed) ...
  "RowNumber": "Row numbers from 1 to 10000", "CustomerId": "Unique IDs for bank customer identification", "Surname": "Customer's last name", "CreditScore": "Credit score of the customer", "Geography": "The country from which the customer belongs", "Gender": "Male or Female", "Age": "Age of the customer", "Tenure": "Number of years for which the customer has been with the bank", "Balance": "Bank balance of the customer", "NumOfProducts": "Number of bank products the customer is utilising", "HasCrCard": "Binary flag for whether the customer holds a credit card with the bank or not", "IsActiveMember": "Binary flag for whether the customer is an active member with the bank or not", "EstimatedSalary": "Estimated salary of the customer in dollars", "Customer ID": "Unique identifier for each customer.", "Item Purchased": "Name of the item purchased.", "Category": "Product category (Clothing, Footwear, Outerwear, Accessories).", "Purchase Amount (USD)": "Total amount spent on the purchase.", "Location": "Geographic location of the customer (city)", "Size": "Size of the item purchased (if applicable; e.g., S, M, L, XL).", "Color": "Color of the item purchased", "Season": "Season of the purchase (e.g., Winter, Summer, Spring, Fall).", "Review Rating": "Customer's review rating for the purchase (e.g., 1-5 stars).", "Subscription Status": "Indicates whether the customer is a subscriber (e.g., Active, Inactive, None).", "Payment Method": "Method used for the payment (e.g., Credit Card, PayPal, Cash, Venmo, Debit Card, Bank Transfer).", "Shipping Type": "Type of shipping chosen (Express, Free Shipping, Next Day Air, Standard, 2-Day Shipping, Store Pickup)", "Discount Applied": "Indicates if a discount was applied (e.g., Yes/No).", "Promo Code Used": "Specifies if a promo code was used during the purchase (e.g., Yes/No).", "Previous Purchases": "Total number of previous purchases made by the customer.", "Preferred Payment Method": "The payment method most frequently used by the customer.", "Frequency of Purchases": "How often the customer makes purchases.", "Date": "The date corresponding to each financial record (from January 2015 onwards).", "Operating_Income": "The income generated from the bank's core business operations.", "Expenses": "Total costs incurred during operations.", "Net_Income": "Profit after subtracting expenses from operating income.", "Assets": "Total assets owned by the bank (e.g., cash, investments).", "Liabilities": "The total debts and obligations owed.", "Equity": "Shareholders' equity, representing the net value of assets minus liabilities.", "Debt_to_Equity": "A financial ratio showing the proportion of debt compared to equity.", "ROA": "A profitability metric calculated as net income divided by total assets.", "Revenue": "Total income from all operations and activities.", "Cash_Flow": "The net cash generated or used in operations.", "Profit_Margin": "A ratio showing the percentage of revenue that remains as profit.", "Interest_Expense": "Costs associated with the bank's borrowings or debts.", "Tax_Expense": "The amount paid as taxes on profits.", "Dividend_Payout": "The portion of earnings distributed to shareholders as dividends.", "Order No": "Unique identifier for the order", "Order Time": "Timestamp when the order was placed", "Order Type": "Type of order (e.g., Dine-in, Takeaway, Delivery)", "Order Taken By": "Name or ID of the staff who took the order", "Customer Name": "Full name of the customer", "Customer Number": "Contact number of the customer", "Item name": "Name of the purchased item", "Quantity": "Number of units of the item ordered", "Item Type": "Category of the item (e.g., Beverage, Main Course, Dessert)", "Price": "Cost of the item before tax and discounts", "Item Tax": "Applicable tax amount on the item", "Item discount": "Discount applied to the item, if any"
};

// --- Helper Components ---

const Alert = ({ children, type = 'info' }) => {
  const bgColors = {
    info: 'bg-blue-50 text-blue-800 border border-blue-200',
    error: 'bg-red-50 text-red-800 border border-red-200',
    success: 'bg-green-50 text-green-800 border border-green-200',
  };
  return <div className={`p-4 rounded-lg ${bgColors[type]} mb-4`}>{children}</div>;
};

const TableDisplay = ({ data }) => {
    if (!data || data.length === 0) return null;
    const headers = Object.keys(data[0]);
    return (
        <div className="overflow-x-auto rounded-lg border border-slate-200 my-4 bg-white"><table className="min-w-full divide-y divide-slate-200"><thead className="bg-slate-50"><tr >{headers.map(header => <th key={header} className="px-4 py-2 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">{header.replace(/_/g, ' ')}</th>)}</tr></thead><tbody className="bg-white divide-y divide-slate-200">{data.map((row, rowIndex) => (<tr key={rowIndex}>{headers.map(header => <td key={`${rowIndex}-${header}`} className="px-4 py-2 whitespace-nowrap text-sm text-slate-700">{String(row[header])}</td>)}</tr>))}</tbody></table></div>
    );
};

// --- MODIFIED InsightCard component ---
const InsightCard = ({ recommendation, onClick }) => {
  // Derive summary data from the full recommendation object
  const description = recommendation.finding?.text?.split('.')[0] + '.' || "Click to see details.";
  
  return (
    <div
      className="bg-white rounded-xl p-6 shadow-lg border border-slate-200 hover:shadow-xl transition-all duration-300 cursor-pointer group hover:-translate-y-1"
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="w-12 h-12 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-lg flex items-center justify-center group-hover:from-blue-200 group-hover:to-indigo-200 transition-colors">
            <BarChart3 className="w-6 h-6 text-blue-600" />
          </div>
          <h3 className="font-semibold text-slate-800 group-hover:text-blue-600 transition-colors">
            {recommendation.title}
          </h3>
        </div>
      </div>
      <p className="text-slate-600 mb-4 leading-relaxed">{description}</p>
      {/* The green impact circle has been removed */}
      <div className="flex items-center justify-end">
        <span className="text-sm text-blue-600 font-medium group-hover:text-blue-700">
          Click to expand →
        </span>
      </div>
    </div>
  );
};


// --- MODIFIED RecommendationModal component ---
const RecommendationModal = ({ recommendation, sessionId, onClose, onNewRecommendation }) => {
    if (!recommendation) return null;

    // The logic to find chartData and the chart itself have been removed.

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4" onClick={onClose}>
            <div className="bg-slate-50 rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
                <div className="sticky top-0 bg-slate-50/80 backdrop-blur-sm p-6 border-b border-slate-200 z-10 flex justify-between items-center">
                    <h2 className="text-2xl font-bold text-slate-800">{recommendation.title}</h2>
                    <button onClick={onClose} className="text-slate-500 hover:text-slate-800"><X /></button>
                </div>
                
                <div className="p-6 space-y-6">
                    {/* The Chart Display has been removed from here. */}

                    {/* Re-using OLD components inside the new modal */}
                    <RecommendationSection title="Finding" section={recommendation.finding} />
                    <RecommendationSection title="Action Logic" section={recommendation.action_logic} />
                    <RecommendationSection title="Implementation Feasibility" section={recommendation.feasibility} />
                    <RecommendationSection title="Expected Effect" section={recommendation.effect} />
                    <ChatInterface 
                        recommendationId={recommendation.id} 
                        sessionId={sessionId}
                        onNewRecommendation={onNewRecommendation}
                    />
                </div>
            </div>
        </div>
    );
}

// --- These components are now used inside the Modal ---
const RecommendationSection = ({ title, section }) => {
    const [isOpen, setIsOpen] = useState(true);
    if (!section || !section.text) return null;
    return (
        <div className="bg-white border border-slate-200 rounded-lg p-4">
            <button onClick={() => setIsOpen(!isOpen)} className="w-full flex justify-between items-center text-left text-lg font-semibold text-slate-700 hover:text-blue-600 transition-colors">
                <span>{title}</span><ChevronDown className={`transform transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>
            {isOpen && (<div className="mt-3 pl-2 border-l-2 border-blue-100"><div className="prose prose-sm max-w-none text-slate-800"><ReactMarkdown>{section.text}</ReactMarkdown></div><TableDisplay data={section.table_data} /></div>)}
        </div>
    );
};

const ChatInterface = ({ recommendationId, sessionId, onNewRecommendation }) => {
    const [input, setInput] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [isSending, setIsSending] = useState(false);
    const [error, setError] = useState('');

    const handleChatSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || !sessionId) return;
        
        setIsSending(true);
        setError('');

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    feedback: input,
                    recommendation_id: recommendationId,
                }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to get response.');

            onNewRecommendation(data.new_recommendation);
            if (data.chat_history_update) {
                setChatHistory(data.chat_history_update.history);
            }
            setInput('');
        } catch (err) {
            setError(err.message);
        } finally {
            setIsSending(false);
        }
    };
    
    return (
        <div className="bg-white border border-slate-200 rounded-lg p-4">
            <h4 className="text-md font-semibold text-slate-700 mb-2 flex items-center gap-2"><MessageSquare size={18} /> Chat & Refine</h4>
            <div className="bg-slate-50 p-3 rounded-lg max-h-48 overflow-y-auto mb-2 space-y-3">
                {chatHistory.length === 0 ? (<p className="text-sm text-slate-500 text-center">Ask a question or provide feedback to generate a new, refined recommendation.</p>) : (
                    chatHistory.map((msg, idx) => (
                        <div key={idx} className="flex gap-2 items-start">
                           {msg.user ? <User className="w-5 h-5 text-blue-600 flex-shrink-0 mt-1" /> : <Bot className="w-5 h-5 text-green-600 flex-shrink-0 mt-1" />}
                           <div className="text-sm">
                              <p className="font-bold">{msg.user ? "You" : "Assistant"}</p>
                              <p>{msg.user || msg.assistant}</p>
                           </div>
                        </div>
                    ))
                )}
            </div>
            <form onSubmit={handleChatSubmit} className="flex gap-2">
                <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="e.g., 'Analyze this by Order Type instead...'" className="flex-grow p-2 border rounded-md text-sm" disabled={isSending}/>
                <button type="submit" className="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-md hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 flex items-center gap-2" disabled={isSending}>
                    <Send size={16}/> {isSending ? 'Sending...' : 'Send'}
                </button>
            </form>
            {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
        </div>
    );
};

// --- Main App Component (with new look and feel) ---
const DataAnalysisApp = () => {
  const [file, setFile] = useState(null);
  const [description, setDescription] = useState('');
  const [columnsDescriptions, setColumnsDescriptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [recommendations, setRecommendations] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [sessionId, setSessionId] = useState(null);
  const [iterations, setIterations] = useState(2);
  const [selectedRec, setSelectedRec] = useState(null); // For the modal

  useEffect(() => { checkBackendConnection(); }, []);

  const checkBackendConnection = async () => {
    try {
      const response = await fetch('/api/analyze', { method: 'HEAD' });
      setBackendStatus(response.ok ? 'connected' : 'error');
    } catch (err) {
      setBackendStatus('error');
      setError(`Cannot connect to backend. Please ensure the server is running.`);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onload = (event) => {
        const firstLine = event.target.result.split('\n')[0];
        const headers = firstLine.split(',').map(h => h.trim().replace(/"/g, ''));
        const initialDescriptions = headers.map(header => ({
          name: header, description: COLUMNS_DICT[header] || ''
        }));
        setColumnsDescriptions(initialDescriptions);
      };
      reader.readAsText(selectedFile);
    }
  };

  const handleColumnDescriptionChange = (index, description) => {
    setColumnsDescriptions(prev => prev.map((item, i) => i === index ? {...item, description} : item));
  };

  const formatColumnDescriptions = () => {
    return columnsDescriptions
      .filter(col => col.description.trim() !== '')
      .map(col => `${col.name}: ${col.description}`)
      .join('\n');
  };
  
  const handleNewRecommendation = (newRec) => {
    setRecommendations(prevRecs => [...prevRecs, newRec]);
    setSelectedRec(newRec); // Automatically open the new recommendation in the modal
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setRecommendations(null);
    setSelectedRec(null);
    setSessionId(null);
    
    try {
      if (!file) throw new Error('Please select a CSV file');
  
      const formData = new FormData();
      formData.append('file', file);
      formData.append('description', description);
      formData.append('columns', formatColumnDescriptions());
      formData.append('iterations', iterations.toString());
  
      // UPDATED: Add timeout handling
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
        // Increase client timeout to 5 minutes
        signal: AbortSignal.timeout(300000)
      });
      
      const data = await response.json();
      
      if (!response.ok) throw new Error(data.error || 'Analysis failed');
  
      setSessionId(data.session_id);
      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="min-h-screen bg-slate-50 font-sans">
        {/* New Header */}
        <header className="bg-white/80 backdrop-blur-sm border-b border-slate-200 sticky top-0 z-40">
            <div className="container mx-auto px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
                            <BarChart3 className="w-5 h-5 text-white" />
                        </div>
                        <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                            DataWise Agent
                        </h1>
                    </div>
                </div>
            </div>
        </header>

        {/* Main Content */}
        <main className="container mx-auto px-6 py-8">
            <div className="max-w-6xl mx-auto">
                <div className="text-center mb-8">
                    <h2 className="text-4xl font-bold text-slate-800 mb-2">
                        Turn Data into Actionable Insights
                    </h2>
                    <p className="text-lg text-slate-600 max-w-2xl mx-auto">
                        Upload your file, describe your data, and get deep-dive recommendations.
                    </p>
                </div>

                {/* Re-styled Form */}
                <form onSubmit={handleSubmit} className="space-y-6 bg-white p-8 rounded-2xl shadow-xl border border-slate-200 mb-12">
                  <div className="grid md:grid-cols-2 gap-6">
                    {/* Section 1: Upload & Describe */}
                    <div className="space-y-4">
                       <div>
                         <label className="text-lg font-semibold text-slate-700 flex items-center gap-2 mb-2"><Upload size={20}/>1. Upload & Describe</label>
                         <input type="file" accept=".csv" onChange={handleFileChange} className="w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" required />
                         {file && <p className="text-sm text-green-600 mt-2">Selected: {file.name}</p>}
                       </div>
                       <div>
                         <textarea value={description} onChange={(e) => setDescription(e.target.value)} className="w-full h-24 p-2 border rounded-md text-sm" placeholder="e.g., This dataset contains monthly sales data..." required />
                       </div>
                    </div>
                    {/* Section 2: Configure */}
                    <div className="space-y-4">
                      <div>
                        <label className="text-lg font-semibold text-slate-700 flex items-center gap-2 mb-2"><Settings size={20}/>2. Configure Analysis</label>
                        <div className="flex items-center">
                          <input type="range" min="1" max="5" value={iterations} onChange={(e) => setIterations(parseInt(e.target.value))} className="w-full mr-3" />
                          <span className="bg-blue-100 px-3 py-1 rounded-full text-blue-800 font-medium text-sm">{iterations}</span>
                        </div>
                        <p className="text-sm text-slate-500 mt-1">Number of distinct recommendations to generate.</p>
                      </div>
                    </div>
                  </div>

                  {/* Optional Column Descriptions */}
                  {columnsDescriptions.length > 0 && (
                      <div>
                          <h3 className="text-lg font-semibold text-slate-700 mb-2">3. Describe Columns (Optional but Recommended)</h3>
                          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-4 max-h-60 overflow-y-auto p-2 bg-slate-50 rounded-lg">
                              {columnsDescriptions.map((col, index) => (
                                  <div key={col.name}><label className="font-medium text-sm mb-1">{col.name}</label><textarea value={col.description} onChange={(e) => handleColumnDescriptionChange(index, e.target.value)} className="w-full p-2 border rounded-md text-sm" placeholder={`Describe ${col.name}...`} rows={1} /></div>
                              ))}
                          </div>
                      </div>
                  )}

                  <button type="submit" disabled={loading || backendStatus !== 'connected' || !file} className="w-full mt-4 py-3 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-lg font-semibold flex items-center justify-center gap-2">
                      {loading ? 'Analyzing...' : 'Run Analysis'}
                  </button>
                </form>
                
                {/* Status Messages */}
                {error && <Alert type="error">{error}</Alert>}
                {loading && <div className="text-center py-8"><div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mx-auto"></div><p className="mt-4 text-lg text-slate-600">Analyzing data, please wait...</p></div>}
                
                {/* --- NEW Results Display --- */}
                {recommendations && recommendations.length > 0 && (
                    <div>
                        <h2 className="text-3xl font-bold text-slate-800 mb-2">Analysis Complete</h2>
                        <p className="text-slate-600 mb-6">Here are your key insights. Click any card to see the full details and provide feedback.</p>
                        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {recommendations.map((rec) => (
                                <InsightCard 
                                    key={rec.id} 
                                    recommendation={rec} 
                                    onClick={() => setSelectedRec(rec)} 
                                />
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </main>
        
        {/* Modal display logic */}
        {selectedRec && (
            <RecommendationModal
                recommendation={selectedRec}
                sessionId={sessionId}
                onClose={() => setSelectedRec(null)}
                onNewRecommendation={handleNewRecommendation}
            />
        )}
      </div>
    </>
  );
};

export default DataAnalysisApp;