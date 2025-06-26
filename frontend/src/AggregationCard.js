import React, { useState } from 'react';
import { Database, ArrowRight, BarChart, FileText, MessageCircle } from 'lucide-react';
import AggregationChat from './AggregationChat';

// Table Component for displaying aggregation tables
const DataTable = ({ data }) => {
  if (!data || data.length === 0) {
    return <div className="text-gray-500 italic">No table data available</div>;
  }

  // Get all columns from the first row
  const columns = Object.keys(data[0]);

  return (
    <div className="overflow-x-auto bg-white rounded-lg shadow">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((column, idx) => (
              <th 
                key={idx} 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.map((row, rowIdx) => (
            <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
              {columns.map((column, colIdx) => (
                <td 
                  key={colIdx} 
                  className="px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                >
                  {row[column] !== null && row[column] !== undefined ? row[column].toString() : ''}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Markdown renderer component
const MarkdownRenderer = ({ content }) => {
  if (!content) return null;

  // Convert markdown headings
  const withHeadings = content.replace(/^### (.*$)/gm, '<h3 class="text-lg font-semibold mt-4 mb-2">$1</h3>')
    .replace(/^## (.*$)/gm, '<h2 class="text-xl font-semibold mt-6 mb-3">$1</h2>')
    .replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold mt-6 mb-4">$1</h1>');
  
  // Convert markdown lists
  const withLists = withHeadings
    .replace(/^\s*\n\* (.*$)/gm, '<ul class="list-disc pl-5 my-2"><li>$1</li></ul>')
    .replace(/^\s*\n- (.*$)/gm, '<ul class="list-disc pl-5 my-2"><li>$1</li></ul>')
    .replace(/^\s*\n\d\. (.*$)/gm, '<ol class="list-decimal pl-5 my-2"><li>$1</li></ol>');
  
  // Convert paragraphs
  const withParagraphs = withLists
    .split(/\n\s*\n/)
    .map(paragraph => `<p class="my-2">${paragraph}</p>`)
    .join('');

  return (
    <div 
      className="prose prose-blue max-w-none" 
      dangerouslySetInnerHTML={{ __html: withParagraphs }} 
    />
  );
};

// AggregationCard Component to display individual aggregation results
const AggregationCard = ({ 
  aggregation, 
  index, 
  sessionId, 
  onNewAggregation 
}) => {
  const [expanded, setExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState('analysis'); // 'analysis' or 'chat'

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden mb-6 transition-shadow hover:shadow-xl">
      {/* Header */}
      <div className="bg-blue-50 p-4 border-b border-blue-100">
        <div className="flex items-start justify-between">
          <div className="flex items-center">
            <div className="bg-blue-100 rounded-full p-2 mr-3">
              <Database className="h-5 w-5 text-blue-600" />
            </div>
            <h3 className="text-xl font-semibold text-blue-800">
              {index}. {aggregation.title}
            </h3>
          </div>
          <button 
            onClick={() => setExpanded(!expanded)}
            className="text-blue-600 hover:text-blue-800 transition-colors"
          >
            {expanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
        <p className="text-gray-700 mt-2">{aggregation.description}</p>
        
        {/* Show base feedback if this aggregation was generated from feedback */}
        {aggregation.based_on_feedback && (
          <div className="mt-2 p-2 bg-yellow-50 border border-yellow-100 rounded-md">
            <p className="text-sm text-yellow-800">
              <span className="font-semibold">Generated from feedback:</span> {aggregation.based_on_feedback}
            </p>
          </div>
        )}
      </div>

      {/* Expandable Content */}
      {expanded && (
        <div className="p-4">
          {/* Approach Section */}
          <div className="mb-6">
            <h4 className="text-lg font-medium text-gray-800 mb-2 flex items-center">
              <ArrowRight className="w-4 h-4 mr-1 text-blue-500" />
              Approach
            </h4>
            <div className="pl-6 border-l-2 border-blue-100 text-gray-600">
              {aggregation.approach.split('\n').map((line, idx) => (
                <p key={idx} className="mb-1">{line}</p>
              ))}
            </div>
          </div>

          {/* Tabs for Analysis and Chat */}
          <div className="border-b border-gray-200 mb-4">
            <div className="flex space-x-4">
              <button
                onClick={() => setActiveTab('analysis')}
                className={`py-2 px-4 border-b-2 font-medium text-sm ${
                  activeTab === 'analysis' 
                    ? 'border-blue-500 text-blue-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center">
                  <FileText className="w-4 h-4 mr-2" />
                  Analysis & Recommendations
                </div>
              </button>
              <button
                onClick={() => setActiveTab('chat')}
                className={`py-2 px-4 border-b-2 font-medium text-sm ${
                  activeTab === 'chat' 
                    ? 'border-blue-500 text-blue-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center">
                  <MessageCircle className="w-4 h-4 mr-2" />
                  Chat & Feedback
                </div>
              </button>
            </div>
          </div>

          {/* Data Table Section - Always visible */}
          {aggregation.table_data && (
            <div className="mb-6">
              <h4 className="text-lg font-medium text-gray-800 mb-2 flex items-center">
                <BarChart className="w-4 h-4 mr-1 text-blue-500" />
                Data Table
              </h4>
              <DataTable data={aggregation.table_data} />
            </div>
          )}

          {/* Tab Content */}
          {activeTab === 'analysis' ? (
            // Analysis Section
            <div className="mb-6">
              <h4 className="text-lg font-medium text-gray-800 mb-2 flex items-center">
                <FileText className="w-4 h-4 mr-1 text-blue-500" />
                Analysis and Recommendations
              </h4>
              <div className="bg-gray-50 p-4 rounded-lg">
                <MarkdownRenderer content={aggregation.analysis} />
              </div>
            </div>
          ) : (
            // Chat Section
            <div className="mb-6">
              <h4 className="text-lg font-medium text-gray-800 mb-2 flex items-center">
                <MessageCircle className="w-4 h-4 mr-1 text-blue-500" />
                Chat & Feedback
              </h4>
              <div className="bg-gray-50 rounded-lg overflow-hidden" style={{ minHeight: '300px' }}>
                <AggregationChat
                  sessionId={sessionId}
                  aggregation={aggregation}
                  onNewAggregation={onNewAggregation}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AggregationCard;