import React,{useState} from 'react';
import Uploader from '../../Components/Uploader';
import './extract.css';

export default function Extract() {

  const [watermarkImage, setwatermarkImage] = useState(null);

  const handleExtract=()=>{
    if(!watermarkImage){
      alert('Please upload an image');
      return;
    }
  }

  return (
    <div className='container'>
      <Uploader text="Watermarked Image" formId="extract"/>
      <button style={{marginLeft:530}} onClick={handleExtract}>Embed Image</button>
    </div>
    
  )
}
