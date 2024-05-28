
import React, { useState } from 'react';
import axios from 'axios';
import { Link, NavLink } from "react-router-dom";
function ImageUpload() {
    const [selectedImage, setSelectedImage] = useState(null);
    const [processedImage, setProcessedImage] = useState(null);
    const [groundImage, setGroundImage] = useState(null);
    const [selectedImageUrl, setSelectedImageUrl] = useState(null);
    // ///////////////code for evaluation part /////////////////

    const [psnr, setPsnr] = useState(0);
    const [ssim, setSsim] = useState(0);

    ////////////////////////////////////////////////////////////////////

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        setSelectedImage(file);
        const imageUrl = URL.createObjectURL(file);
        setSelectedImageUrl(imageUrl);
    };

    const handleUpload = () => {
        const formData = new FormData();
        formData.append('image', selectedImage);

        axios.post('http://localhost:5000/process_image', formData, {
            responseType: 'json'
        })
            .then(response => {
                setProcessedImage(response.data.processed_image_url);
                setGroundImage(response.data.ground_image_url);
                setPsnr(response.data.psnr);
                setSsim(response.data.ssim);
            })
            .catch(error => {
                console.error('Error uploading image:', error);
            });
    };

    return (
        <>

            <div className=' flex flex-col justify-center items-center gap-5 p-10'>
                <h2 className='text-4xl text-black'>Implementation</h2>
                <input type="file" accept="image/*" onChange={handleImageChange} className='p-2 border border-yellow-300 w-fit text-black font-semibold bg-gray-200 rounded-xl' />
                <button onClick={handleUpload} className='bg-blue-600 px-6 py-3 rounded-xl text-white text-xl font-semibold hover:bg-blue-500'>Upload</button>

                <div className='flex gap-5'>
                    {selectedImageUrl &&
                        <div className=' flex flex-col image-container'>
                            <img src={selectedImageUrl} alt="Selected/Input Image" className='processed-image' />
                            <h3>Input Image</h3>
                        </div>

                    }
                    {processedImage &&
                        <div className="flex flex-col image-container">
                            <img src={processedImage} alt="Processed Image" className="processed-image" />
                            <h3>Output Image</h3>
                        </div>
                    }
                    {groundImage &&
                        <div className="flex flex-col image-container">
                            <img src={groundImage} alt="Ground Image" className="processed-image" />
                            <h3>Ground Truth</h3>
                        </div>
                    }
                </div>
                <div>
                    <h2>Evaluation metrics for above Image</h2>
                    <div>PSNR : {psnr.toFixed(2)}</div>
                    <div>SSIM : {ssim.toFixed(2)}</div>
                </div>
            </div>

        </>


    );
}

export default ImageUpload;

