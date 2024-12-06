import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import './extract.css';

const Extract = () => {
  const [coverImage, setCoverImage] = useState(null);
  const [secretImage, setSecretImage] = useState(null);
  const [embedResult, setEmbedResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelect = (file, formId) => {
    if (formId === 'cover') {
      setCoverImage(file);
    } else if (formId === 'embed') {
      setSecretImage(file);
    }
  };

  const handleEmbed = () => {
    if (!coverImage || !secretImage) {
      setError('Please upload both images');
      return;
    }

    const formData = new FormData();
    formData.append('image1', coverImage);
    formData.append('image2', secretImage);

    axios
      .post('http://127.0.0.1:8000/xray/', formData, {
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
        setError('Error embedding image.');
      });
  };

  const downloadImage = () => {
    const link = document.createElement('a');
    link.href = embedResult;
    link.download = 'd-image.jpg';
    link.click();
  };

  return (
    <div className="bg-[#161b22] bg-custom-bg min-h-screen flex justify-center items-center pt-10">
      <div className="backdrop-blur-sm bg-white/10 rounded-lg shadow-lg p-8 w-full max-w-4xl ml-64">
        <h1 className="text-white text-4xl font-bold mb-4 ml-32">
          What Would You Like To Extract?
        </h1>

        <div className='flex space-x-20 justify-center'>
          <Link to='/xrayex'><button className='button'>Signature in X-ray</button></Link>
          <Link to='/docex'><button className='button'>Logo in Document</button></Link>
        </div>
       </div> 
    </div>
  );
};

export default Extract;