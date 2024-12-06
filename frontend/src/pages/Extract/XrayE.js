import React, { useState } from 'react';
import axios from 'axios';
import Uploader from '../../Components/Uploader';
//import './xray.css';

const XrayE = () => {
  const [coverImage, setCoverImage] = useState(null);
  //const [secretImage, setSecretImage] = useState(null);
  const [embedResult, setEmbedResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelect = (file, formId) => {
    if (formId === 'cover') {
      setCoverImage(file);
    }
  };

  const handleEmbed = () => {
    if (!coverImage) {
      setError('Please upload both images');
      return;
    }

    const formData = new FormData();
    formData.append('image', coverImage);

    axios
      .post('http://127.0.0.1:8000/xrayExtract/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      .then((response) => {
        if (response.data.image) {
          setEmbedResult(response.data.image);
        } else {
          setError('Error: No image data received');
        }
      })
      .catch((error) => {
        console.error('Error embedding image:', error);
        setError('Error extracting image.');
      });
  };

  const downloadImage = () => {
    const link = document.createElement('a');
    link.href = embedResult;
    link.download = 'd-image.jpg';
    link.click();
  };

  return (
    <div className="bg-[#161b22] bg-custom-radial min-h-screen flex justify-center items-center pt-10">
      <div className="backdrop-blur-sm bg-white/10 rounded-lg shadow-lg p-8 w-full max-w-4xl ml-64">
        <h1 className='text-2xl text-white ml-[200px]'>Select your Xray watermarked image</h1>
        <br />
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="w-full md:w-1/2 mb-8 md:mb-0 mr-5 ml-[200px]">
            <Uploader
              text="Watermarked image"
              formId="cover"
              onImageSelect={(file) => handleImageSelect(file, 'cover')}
            />
          </div>
        </div>

        <button
          onClick={handleEmbed}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded block mx-auto mt-8"
        >
          Extract Image
        </button>

        {embedResult && (
          <div className="mt-8">
            <img
              src={embedResult}
              alt="Extracted Result"
              className="max-w-full h-auto mb-4 block mx-auto"
            />
            <button
              onClick={downloadImage}
              className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded block mx-auto"
            >
              Download Image
            </button>
          </div>
        )}

        {error && (
          <div className="text-red-500 font-bold text-center mt-8">{error}</div>
        )}
      </div>
    </div>
  );
};

export default XrayE;