import React, { useState } from 'react';
import axios from 'axios';
import { Button, Box, Typography } from '@mui/material';
import { styled } from '@mui/system';

const Input = styled('input')({
  display: 'none',
});

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
    <Box sx={{ '& > :not(style)': { m: 1 } }}>
      <label htmlFor="contained-button-file">
        <Input accept="image/*" id="contained-button-file" type="file" onChange={onFileChange} />
        <Button variant="contained" component="span">
          Select Image
        </Button>
      </label>
      <Button variant="contained" color="primary" onClick={onFileUpload}>
        Upload
      </Button>
      {results.map((result, index) => (
        <Box key={index} sx={{ my: 2 }}>
          <Typography variant="h6">Result {index + 1}</Typography>
          <Typography variant="body1">Image Path: {result.image_path}</Typography>
          <Typography variant="body1">Image Link: {result.image_link}</Typography>
          <Typography variant="body1">Matches: {result.matches}</Typography>
        </Box>
      ))}
    </Box>
  );
}

export default ImageUpload;
