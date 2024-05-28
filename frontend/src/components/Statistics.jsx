import React from 'react'

function Statistics() {
  return (
    <>
      <div>
        <div className='container pb-3 text-black  text-2xl font-display flex flex-col gap-4'>
          <h2 className='font-extrabold'>PSNR (Peak Signal-to-Noise Ratio)</h2>
          <p>It tells you how much the processed image differs from the original due to noise, with higher values indicating less noise and better quality.</p>
        </div>

        <div className='container pb-3 mt-1 text-black  text-2xl font-display flex flex-col gap-4'>
          <h2 className='font-extrabold'>SSIM  (Structural Similarity Index)</h2>
          <p>It's a method used to measure the similarity between two images, assessing how close they are in terms of structure, luminance, and contrast. SSIM gives a score between -1 and 1, where 1 means the images are exactly the same. It's often used in image processing to evaluate how well an image processing algorithm preserves the quality and details of an image.</p>
        </div>

       
        <div id="graphs" className='container h-500 flex flex-col justify-center items-center gap-12'>
          <h2 className='font-extrabold text-3xl'>Graphs</h2>
          <div className='flex flex-col justify-center items-center' >
            <img src=".\src\assets\ssim_graph.jpg" alt=""  />
            <p>SSIM  v/s Epochs</p>
        </div>
          <div className='w-2/4 flex flex-col justify-center items-center'>
            <img src=".\src\assets\psnr_graph.jpg" alt=""  />
            <p>PSNR  v/s Epochs</p>
        </div>
          <div className='w-2/4 flex flex-col justify-center items-center'>
            <img src=".\src\assets\loss_graph.jpg" alt=""  />
            <p>Loss  v/s Epochs</p>
        </div>
          
        </div>
      </div>
    </>
  )
}

export default Statistics