import React, { useState } from 'react';
import './uploader.css';
import { MdCloudUpload, MdDelete } from 'react-icons/md';
import { AiFillFileImage } from 'react-icons/ai';

function Uploader({ formId, onImageSelect, text }) {
  const [image, setImage] = useState(null);
  const [fileName, setFileName] = useState('No selected file-');
  const inputFieldClass = `input-field-${formId}`;

  return (
    <main className="uploader-container">
      <form
        onClick={() => document.querySelector('.' + inputFieldClass).click()}
      >
        <input
          type="file"
          accept='image/*'
          className={inputFieldClass}
          hidden
          onChange={({ target: { files } }) => {
            if (files[0]) {
              setFileName(files[0].name);
              setImage(URL.createObjectURL(files[0]));
              onImageSelect(files[0]); // Pass the file up to the parent component
            }
          }}
        />
        <div className="image-wrapper">
          {image ? (
            <img src={image} alt={fileName} />
          ) : (
            <>
              <MdCloudUpload color='purple' size={60} />
              <p>Upload {text}</p>
            </>
          )}
        </div>
      </form>
      <section>
        <AiFillFileImage color='purple' />
        <span style={{ color: 'whitesmoke', display:'flex' }}>
          {fileName}
          <MdDelete
            color='purple'
            size={20}
            onClick={() => {
              setImage(null);
              setFileName('No selected file');
              onImageSelect(null); // Clear the file in parent component
            }}
          />
        </span>
      </section>
    </main>
  );
}

export default Uploader;
