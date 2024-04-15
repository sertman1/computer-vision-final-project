import React, { useState } from 'react';
import axios from 'axios';

const ImageUpload = () => {
  const [results, setResults] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);

  const onFileChange = event => {
    setSelectedFile(event.target.files[0]);
  };

  const onFileUpload = () => {
    const formData = new FormData();
    formData.append('file', selectedFile);
    axios.post('http://localhost:5000/upload', formData)
      .then(response => {
        setResults(response.data);
      })
      .catch(error => {
        console.error(error);
      });
  };

  return (
    <div>
      <input type='file' onChange={onFileChange} />
      <button onClick={onFileUpload}>Upload!</button>
      {results.map((result, index) => (
        <div key={index}>
          <p>Image Path: {result.image_path}</p>
          <p>Image Link: {result.image_link}</p>
          <p>Matches: {result.matches}</p>
        </div>
      ))}
    </div>
  );
}

export default ImageUpload;
