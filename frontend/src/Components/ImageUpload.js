import React, { useState } from 'react';
import axios from 'axios';
import { Button, Box, Typography, ThemeProvider, createTheme } from '@mui/material';
import { styled } from '@mui/system';

const Input = styled('input')({
  display: 'none',
});

const theme = createTheme({
  palette: {
    primary: {
      main: '#ff9800', // orange
    },
    secondary: {
      main: '#9e9e9e', // grey
    },
    background: {
      default: '#ffffff', // white
    },
  },
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
    <ThemeProvider theme={theme}>
      <Box sx={{ bgcolor: 'background.default', height: '100vh', '& > :not(style)': { m: 1 } }}>
        <label htmlFor="contained-button-file">
          <Input accept="image/*" id="contained-button-file" type="file" onChange={onFileChange} />
          <Button variant="contained" component="span" color="primary">
            Select Image
          </Button>
        </label>
        <Button variant="contained" color="secondary" onClick={onFileUpload}>
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
    </ThemeProvider>
  );
}

export default ImageUpload;
