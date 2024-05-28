import React from 'react'

function Methodology() {
  return (
    <>
      <div className="container  flex flex-col items-center  gap-6">
        <h2 className='font-display text-3xl font-extrabold'>CNN Model Architecture</h2>
        <p className='font-medium text-2xl'>
          Our model enhances underwater images using a convolutional neural network (CNN) with a U-Net architecture. The U-Net model, consisting of downsampling and upsampling paths with skip connections, is built using Keras. The model is trained with mean squared error loss and metrics like PSNR and SSIM, using TensorBoard for logging.  Our model leverages deep learning for effective underwater image enhancement.
        </p>
        <h2 className='font-display text-3xl font-extrabold'>Post Processing</h2>
        <p className='font-medium text-2xl'>
          The code enhances underwater images by balancing colors, improving contrast, and adjusting brightness and saturation. It first equalizes the RGB channels to ensure even color distribution, then stretches the histogram of each channel to enhance contrast, and finally adjusts saturation and brightness using HSV stretching. This process is applied to the input image using specified parameters to test different enhancement settings, and the enhanced image is returned.
        </p>
        <h2 className='font-display text-3xl font-extrabold'>Model Architecture</h2>
        <img src=".\src\assets\Model_Arch.png" alt="" className='w-2/4' />
        
        
      </div>
    </>
  )
}

export default Methodology