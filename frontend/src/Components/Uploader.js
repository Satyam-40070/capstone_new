import React, { useState } from 'react';

function Uploader({ formId, onImageSelect, text = 'Image' }) {
  const [image, setImage] = useState(null);
  const [fileName, setFileName] = useState('No selected file');
  const inputFieldClass = `input-field-${formId}`;

  const handleFileChange = (files) => {
    if (files[0]) {
      setFileName(files[0].name);
      setImage(URL.createObjectURL(files[0]));
      onImageSelect(files[0]);
    }
  };

  const clearImage = () => {
    setImage(null);
    setFileName('No selected file');
    onImageSelect(null);
  };

  return (
    <div className="flex flex-col items-center justify-center w-full max-w-md mx-auto p-4 bg-blue-100 rounded-lg shadow-md hover:shadow-xl transition-all duration-300">
      <form 
        className="w-full cursor-pointer"
        onClick={() => document.querySelector('.' + inputFieldClass).click()}
      >
        <input
          type="file"
          accept="image/*"
          className={`${inputFieldClass} hidden`}
          onChange={({ target: { files } }) => handleFileChange(files)}
        />
        
        <div className="w-full h-64 border-2 border-dashed border-purple-300 rounded-lg flex flex-col items-center justify-center hover:border-purple-500 transition-colors">
          {image ? (
            <img 
              src={image} 
              alt={fileName} 
              className="w-full h-full object-cover rounded-lg"
            />
          ) : (
            <div className="flex flex-col items-center text-center">
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className="h-12 w-12 text-purple-500 mb-2" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
                />
              </svg>
              <p className="text-gray-600 mt-2">Upload {text}</p>
            </div>
          )}
        </div>
      </form>
      
      <div className="w-full mt-4 flex items-center justify-between bg-gray-100 p-3 rounded-lg">
        <div className="flex items-center space-x-2">
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="h-5 w-5 text-purple-500" 
            viewBox="0 0 20 20" 
            fill="currentColor"
          >
            <path 
              fillRule="evenodd" 
              d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" 
              clipRule="evenodd" 
            />
          </svg>
          <span className="text-sm text-gray-700 truncate max-w-[200px]">
            {fileName}
          </span>
        </div>
        
        {image && (
          <button 
            onClick={clearImage} 
            className="text-purple-500 hover:text-purple-700 transition-colors"
            aria-label="Delete image"
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              className="h-5 w-5" 
              viewBox="0 0 20 20" 
              fill="currentColor"
            >
              <path 
                fillRule="evenodd" 
                d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" 
                clipRule="evenodd" 
              />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}

export default Uploader;