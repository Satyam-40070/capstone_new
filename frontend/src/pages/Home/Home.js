import React,{useState} from 'react';
import axios from 'axios';
//import { useToPng } from '@hugocxl/react-to-image';
import './Home.css';
import Uploader from '../../Components/Uploader';

export default function Home() {

  /*const [textInput, setTextInput] = useState('');
  const [convertedImage, setConvertedImage] = useState(null);

  const handleTextChange = (e) => {
    setTextInput(e.target.value);
  };

  const convertTextToImage = () => {
    // Convert text to image using html-to-image library
    reactToImage.toPng(document.getElementById('text-container'))
      .then((dataUrl) => {
        setConvertedImage(dataUrl);
      })
      .catch((error) => {
        console.error('Error converting text to image:', error);
      });
  };*/
  /*const [{ isSuccess }, convert, ref] = useToPng<HTMLDivElement>({
    onSuccess: data => navigator.clipboard.writeText(data)
  })*/
    const [coverImage, setCoverImage] = useState(null);
    const [secretImage, setSecretImage] = useState(null);
    const [embedResult, setEmbedResult] = useState(null); // Add state to store API response
    const [error, setError] = useState(null);
  
    const handleImageSelect = (file, formId) => {
      if (formId === 'cover') {
        setCoverImage(file);
      } else if (formId === 'embed') {
        setSecretImage(file);
      }
    };
 const handleEmbed =()=>{
  if (!coverImage || !secretImage) {
    alert('Please upload both images');
    return;
  }

  const formData = new FormData();
  formData.append('image1', coverImage);
  formData.append('image2', secretImage);
  console.log(formData.get('coverImage'))


  axios.post('http://127.0.0.1:8000/resize/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
  .then((response) => {
    if (response.data.image) {
      // Set image data to state or directly display it
      setEmbedResult(response.data.image);
    } else {
      setError('Error: No image data received');
    }
  })
  .catch((error) => {
    console.error('Error embedding image:', error);
    setError('Error embedding image.'); 
  });
}
const downloadImage = () => {
    const link = document.createElement('a');
    link.href = embedResult;
    link.download = 'resized-image.jpg'; // Set the default filename
    link.click();
  };

  return (
    <div className='container1'>
      <div className="content">
      <h1 className='text-3xl'>Securing Digital Communication:</h1><br/>
      <h2>
Advancing Protection with <a href="/" style={{color:'purple'}}>“Image Watermarking”</a> and <a href="/" style={{color:'purple'}}>“Learning Models”</a></h2><br/>
<p> Powered by Deep Learning, We can embed and extract watermarks and objects from photos for desirable purpose. 
  It is also a cross-platform tool available on desktop (Win & Mac), mobile (iOS & Android), and web.</p>


    <Uploader text="Cover Image" formId="cover" onImageSelect={(file) => handleImageSelect(file, 'cover')}/>
      </div>
      
      <div className="image">
        <img src="/waterimg.png" alt="watermark" />
        <Uploader text="Secret Image" formId="embed" onImageSelect={(file) => handleImageSelect(file, 'embed')}/>
      </div>
      <button onClick={handleEmbed}>Embed Image</button>
      {embedResult && (
        <div>
          <img src={embedResult} alt="Embedded Result" style={{ maxWidth: '100%', height: 'auto', marginLeft: '450px', marginBottom:'10px' }} />
          <button onClick={downloadImage} style={{ display: 'block', marginTop: '10px' }}>
            Download Image
          </button>
        </div>
      )}
      
      {error && <div style={{ color: 'red' }}>{error}</div>}
    </div>
  )
}

