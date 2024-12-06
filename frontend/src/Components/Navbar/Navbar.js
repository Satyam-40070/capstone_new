import React, { useState, useEffect } from 'react';

const VerticalNavbar = () => {
  const [isOpen, setIsOpen] = useState(false); // State to manage navbar toggle
  const [activeLink, setActiveLink] = useState('Home'); // State to manage active link

  useEffect(() => {
    // Set initial active link based on current URL
    const currentPath = window.location.pathname;
    const linkMap = {
      '/': 'Home',
      '/extractwatermark': 'Extract Watermark',
      '/trainmodel': 'Train Model',
      '/working': 'Working',
      '/aboutus': 'About Us',
    };
    setActiveLink(linkMap[currentPath] || 'Home'); // Default to 'Home' if path is unrecognized
  }, []);

  const toggleNavbar = () => {
    setIsOpen(!isOpen); // Toggle the navbar open/close
  };

  const handleLinkClick = (item) => {
    setActiveLink(item); // Update the active link
    setIsOpen(false); // Close navbar on mobile
  };

  return (
    <>
      {/* Mobile Menu Toggle */}
      <button
        onClick={toggleNavbar}
        className="fixed top-4 left-4 z-60 md:hidden bg-blue-500 text-white p-2 rounded-lg"
        aria-label={isOpen ? 'Close menu' : 'Open menu'}
      >
        {isOpen ? 'Close' : 'Menu'}
      </button>

      {/* Navbar */}
      <nav
        className={`
          fixed top-0 left-0 h-full w-64 backdrop-blur-sm bg-black/10 shadow-lg z-50 transform transition-transform duration-300
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
          md:translate-x-0
        `}
      >
        <div className="flex flex-col h-full p-6">
          <a href="/" className="mb-10 self-center">
            <img
              src="https://via.placeholder.com/150x50"
              alt="logo"
              className="h-12 w-auto"
            />
          </a>

          <div className="space-y-4 flex-grow">
            {['Home', 'Extract Watermark', 'Train Model', 'Working', 'About Us'].map((item) => (
              <a
                key={item}
                href={item === 'Home' ? '/' : `/${item.toLowerCase().replace(' ', '')}`}
                onClick={() => handleLinkClick(item)}
                className={`block py-3 px-4 rounded-lg transition-all duration-300 ${
                  activeLink === item
                    ? 'bg-blue-100 text-blue-600 font-bold'
                    : 'text-gray-100 hover:bg-blue-50 hover:text-blue-600'
                }`}
              >
                {item}
              </a>
            ))}
          </div>
        </div>
      </nav>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          onClick={toggleNavbar}
          className="fixed inset-0 bg-black opacity-50 z-40 md:hidden"
        />
      )}
    </>
  );
};

export default VerticalNavbar;