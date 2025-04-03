import React, { useState, useRef, useEffect } from 'react';
import { FiMail, FiMapPin } from "react-icons/fi"; 
import { FaFileImage, FaRegFileAlt } from "react-icons/fa";
import { FaBrain, FaBolt, FaRegStar, FaGift } from "react-icons/fa";
import HeroSection from './HeroSection';
import './ImageDeblurring.css';

const ImageDeblurring = () => {
  const [activeTab, setActiveTab] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const [menuOpen, setMenuOpen] = useState(false);

  // Handle file selection
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  // Process the selected file
  const processFile = (file) => {
    // Reset states
    setError(null);
    setProcessedImage(null);
    
    // Validate file type
    if (!file.type.match('image.*')) {
      setError('Please select an image file (JPEG, PNG, etc.)');
      return;
    }

    // Display the original image
    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImage(e.target.result);
      // Call backend API to process the image
      processImageWithBackend(file);
    };
    reader.readAsDataURL(file);
  };

  // Process image with backend ML model
  const processImageWithBackend = async (file) => {
    setIsProcessing(true);
  
    try {
      const formData = new FormData();
      formData.append("image", file);
  
      let endpoint = "";
  
      // Determine the endpoint based on the active tab
      if (activeTab === "text") {
        endpoint = "https://console.cloud.google.com/cloud-build/builds;region=global/00b0b70f-aba2-4ed4-b172-7092a4ba4a4b?hl=en&invt=AbtvUg&project=orbital-bank-455617-v6&supportedpurview=folder/predict/text";
      } else if (activeTab === "general") {
        endpoint = "https://console.cloud.google.com/cloud-build/builds;region=global/00b0b70f-aba2-4ed4-b172-7092a4ba4a4b?hl=en&invt=AbtvUg&project=orbital-bank-455617-v6&supportedpurview=folder/predict/general";
      } else {
        throw new Error("Invalid tab selected");
      }
  
      // Make an API call to the appropriate endpoint
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error("Failed to process image");
      }
  
      // Get the processed image from the response
      const blob = await response.blob();
      const processedImageUrl = URL.createObjectURL(blob);
      setProcessedImage(processedImageUrl);
  
      setIsProcessing(false);
    } catch (err) {
      setError("Error processing image: " + err.message);
      setIsProcessing(false);
    }
  };

  // Handle drag and drop
  useEffect(() => {
    const dropZone = dropZoneRef.current;
    
    const handleDragOver = (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    };
    
    const handleDragLeave = (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    };
    
    const handleDrop = (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
      
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        processFile(e.dataTransfer.files[0]);
      }
    };
    
    if (dropZone) {
      dropZone.addEventListener('dragover', handleDragOver);
      dropZone.addEventListener('dragleave', handleDragLeave);
      dropZone.addEventListener('drop', handleDrop);
      
      return () => {
        dropZone.removeEventListener('dragover', handleDragOver);
        dropZone.removeEventListener('dragleave', handleDragLeave);
        dropZone.removeEventListener('drop', handleDrop);
      };
    }
  }, [dropZoneRef.current]);

  // Handle download of processed image
  const handleDownload = () => {
    if (processedImage) {
      const link = document.createElement('a');
      link.href = processedImage;
      link.download = 'sharpened-image.jpg';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Navbar */}
      <nav className="navbar">
        <div className="navbar-container">
          {/* Logo */}
          <div className="flex items-center">
            <a href='#' className="navbar-logo text-[#55BDC9]">Sharp</a><a href='#' className="navbar-logo text-[#F07C41]">ify</a>
          </div>

          {/* Desktop Menu */}
          <div className={`nav-links ${menuOpen ? 'active' : ''}`}>
            <a href="#features" className="nav-link text-[#c6eef6]">Features</a>
            <a href="#try-it" className="nav-link text-[#c6eef6]">Try It</a>
          </div>

          {/* Mobile Menu Icon */}
          <div className="menu-icon" onClick={() => setMenuOpen(!menuOpen)}>
            {menuOpen ? '✖' : '☰'}
          </div>
        </div>
      </nav>
      
      {/* Hero Section */}
      <HeroSection/> 

      {/* Features Section */}
      <section id="features" className="features-section py-16 bg-gray-900">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <h2 className="features-heading text-white mb-12">Why Choose Our Solution?</h2>

          <div className="features-container grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { title: "Cutting-Edge ML", desc: "Utilizes advanced machine learning for superior image restoration.", icon: <FaBrain className="feature-icon" /> },
              { title: "Lightning Fast", desc: "Processes and enhances images in just seconds.", icon: <FaBolt className="feature-icon" /> },
              { title: "Exceptional Clarity", desc: "Produces ultra-sharp results while retaining intricate details.", icon: <FaRegStar className="feature-icon" /> },
              { title: "Completely Free", desc: "No subscriptions, no hidden costs—just high-quality results.", icon: <FaGift className="feature-icon" /> }
            ].map((feature, i) => (
              <div key={i} className="feature-card transform transition duration-300 hover:scale-105 hover:shadow-2xl">
              <div className="feature-icon-container flex justify-center">
                {feature.icon}
              </div>
              <h3 className="feature-title">{feature.title}</h3>
              <p className="feature-desc">{feature.desc}</p>
            </div>
            ))}
          </div>
        </div>
      </section>


      
      {/* Try It Section */}
      <section id="try-it" className="try-it-section">
        <h2 className="try-it-heading">Try It Now</h2>
        <div className="try-it-container">

          {/* Selection Cards */}
          <div className="flex flex-col md:flex-row justify-center gap-6">
            <div
              className={`cursor-pointer try-it-options try-it-options-text shadow-lg rounded-lg p-6 w-64 flex flex-col items-center transform transition ${
                activeTab === "text" ? "scale-105 shadow-2xl border-2 border-blue-500" : "hover:scale-105"
              }`}
              onClick={() => setActiveTab("text")}
            >
              <FaRegFileAlt className="text-5xl text-green-500 mb-4" />
              <h3 className="text-lg font-semibold text-[#95eefe]">Deblur Text Images</h3>
              <p className="text-[#c6eef6] text-sm mt-2">Enhance scanned documents & handwritten notes</p>
            </div>

            <div
              className={`cursor-pointer try-it-options try-it-options-general shadow-lg rounded-lg p-6 w-64 flex flex-col items-center transform transition ${
                activeTab === "general" ? "scale-105 shadow-2xl border-2 border-green-500" : "hover:scale-105"
              }`}
              onClick={() => setActiveTab("general")}
            >
              <FaFileImage className="text-5xl text-purple-500 mb-4" />
              <h3 className="text-lg font-semibold text-[#95eefe]">Deblur General Images</h3>
              <p className="text-[#c6eef6] text-sm mt-2">Improve photos, portraits & artwork</p>
            </div>
          </div>
          
          {activeTab ? (
            <>
              <div 
                ref={dropZoneRef}
                className="upload-box"
                onClick={() => fileInputRef.current.click()}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  onChange={handleFileSelect}
                  accept="image/*"
                />
                <div className="text-4xl mb-3">📤</div>
                <p className="text-lg text-[#c6eef6] font-semibold">
                  Drop your {activeTab === 'text' ? 'text-containing' : ''} image here or click to browse
                </p>
                <button className="try-it-btn active mt-4">
                  Select File
                </button>
              </div>
              
              {error && (
                <div className="mt-4 p-3 bg-red-200 text-red-800 rounded-lg">
                  {error}
                </div>
              )}
              
              {originalImage && (
                <div className="image-preview-container">
                  <div className="image-box">
                    <h3 className="font-semibold mb-2">Original Image</h3>
                    <div className="h-48 rounded flex items-center justify-center overflow-hidden">
                      <img 
                        src={originalImage} 
                        alt="Original" 
                        className="max-w-full max-h-full object-contain"
                      />
                    </div>
                  </div>
                  <div className="image-box">
                    <h3 className="font-semibold mb-2 text-gray-700">Sharpened Result</h3>
                    <div className="h-48 rounded flex items-center justify-center overflow-hidden">
                      {isProcessing ? (
                        <div className="text-center">
                          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-amber-500 border-t-transparent"></div>
                          <p className="mt-2 text-gray-700">Processing...</p>
                        </div>
                      ) : processedImage ? (
                        <img 
                          src={processedImage} 
                          alt="Processed" 
                          className="max-w-full max-h-full object-contain"
                        />
                      ) : (
                        <span className="text-gray-700">Awaiting processing</span>
                      )}
                    </div>
                  </div>
                </div>
              )}
              
              <div className="mt-6 text-center">
                <button 
                  className="download-btn"
                  disabled={!processedImage}
                  onClick={handleDownload}
                >
                  Download Sharpened Image
                </button>
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-[#c6eef6]">
              Select an option above to deblur your image
            </div>
          )}
        </div>
      </section>

      
      {/* Footer - Clean modern design without dots pattern */}
      <footer className="pt-8 px-6 shadow-2xl relative">
        <div className="container mx-auto max-w-6xl grid grid-cols-1 md:grid-cols-2 gap-64">
            
            <div className="space-y-4 text-left">
              <h2 className="gradient-text text-2xl font-bold mb-4 select-none">
                Sharpify
              </h2>
              <p className="leading-relaxed text-gray-300 mb-4 select-none">
                ML-powered image deblurring that sharpens visuals and restores details, improving quality for photography, surveillance, and more.
              </p>
            </div>
            {/* Contact Information Section */}
            <div className="space-y-6 text-left">
              <div className="flex items-start space-x-4">
                <div className="w-10 h-10 flex items-center justify-center rounded-full bg-slate-800 flex-shrink-0">
                  <FiMapPin size={18} className="text-[#7CFC00]" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg text-[#00BFFF] select-none">Location</h3>
                  <p className="text-gray-300 select-none">
                    Indian Institute of Engineering Science and Technology, Shibpur, West Bengal
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4">
                <div className="w-10 h-10 flex items-center justify-center rounded-full bg-slate-800 flex-shrink-0 ">
                  <FiMail size={18} className="text-red-500" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg text-[#00BFFF] select-none">Email</h3>
                  <a 
                    href="mailto:contact@sharpify.com" 
                    className="transition-colors duration-300 text-gray-200 hover:text-amber-200"
                  >
                    contact@sharpify.com
                  </a>
                </div>
              </div>
            </div>
        </div>

        {/* Divider and Copyright */}
        <div className="text-center mt-6 mb-4 text-gray-400">
          <div className="h-0.5 footer-border w-3/5 mx-auto mb-4 bg-gradient-to-r"></div>
          © {new Date().getFullYear()} Sharpify. All Rights Reserved.
        </div>
      </footer>
    </div>
  );
};

export default ImageDeblurring;