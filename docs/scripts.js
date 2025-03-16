// Initialize the dashboard when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
  // Initialize theme
  initTheme();
  
  // Initialize profile viewer
  setupProfileViewer();
  
  // Setup mobile tab controls
  setupMobileTabControls();
});

// Theme toggle functionality
function initTheme() {
  const themeSwitch = document.getElementById('theme-switch');
  
  // Check for saved theme preference
  const savedTheme = localStorage.getItem('theme');
  
  // If no theme is saved or the saved theme is 'dark', apply dark theme
  if (!savedTheme || savedTheme === 'dark') {
    document.body.classList.add('dark-theme');
    themeSwitch.checked = true;
    
    // Save the theme preference if it wasn't already saved
    if (!savedTheme) {
      localStorage.setItem('theme', 'dark');
    }
  }
  
  // Add event listener for theme switch
  themeSwitch.addEventListener('change', function() {
    if (this.checked) {
      document.body.classList.add('dark-theme');
      localStorage.setItem('theme', 'dark');
    } else {
      document.body.classList.remove('dark-theme');
      localStorage.setItem('theme', 'light');
    }
  });
}

// Add necessary event listeners
function addEventListeners() {
  // Add event listeners for data table interactions
  const viewButtons = document.querySelectorAll('.view-profile-btn');
  viewButtons.forEach(button => {
    button.addEventListener('click', function() {
      const testCase = this.dataset.testCase;
      const model = this.dataset.model;
      
      // Set the dropdown values
      document.getElementById('profile-test-case').value = testCase;
      document.getElementById('profile-model').value = model;
      
      // Trigger profile loading
      document.getElementById('load-profile').click();
      
      // Scroll to profile viewer
      document.querySelector('.profile-viewer-section').scrollIntoView({ behavior: 'smooth' });
    });
  });
}

// Helper function to format test case names for display
function formatTestCaseName(testCase) {
  // Convert camelCase to Title Case with spaces
  return testCase
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, str => str.toUpperCase());
}

// Helper function to convert the data object to an array
function convertDataToArray(data) {
  const result = [];
  
  Object.keys(data).forEach(testCase => {
    Object.keys(data[testCase]).forEach(model => {
      result.push({
        testCase: testCase,
        model: model,
        ...data[testCase][model]
      });
    });
  });
  
  return result;
}

/**
 * Setup mobile tab controls for profile switching
 * This function adds event listeners to the mobile tab buttons
 * and handles the switching between primary and comparison profiles
 */
function setupMobileTabControls() {
  const tabButtons = document.querySelectorAll('.profile-tab-btn');
  const primaryProfile = document.querySelector('.primary-profile');
  const comparisonProfile = document.querySelector('.comparison-profile');
  const compareWithSelect = document.getElementById('compare-with');
  
  // Set initial state - primary profile is active by default
  primaryProfile.classList.add('active');
  
  // Add event listeners to tab buttons
  tabButtons.forEach(button => {
    button.addEventListener('click', function() {
      // First, remove active class from all buttons
      tabButtons.forEach(btn => btn.classList.remove('active'));
      
      // Add active class to clicked button
      this.classList.add('active');
      
      // Get the target profile from data attribute
      const targetProfile = this.getAttribute('data-target');
      
      // Handle visibility of profiles based on target
      if (targetProfile === 'primary-profile') {
        primaryProfile.classList.add('active');
        comparisonProfile.classList.remove('active');
      } else if (targetProfile === 'comparison-profile') {
        // Check if there's a comparison selected
        if (compareWithSelect.value) {          
          // Toggle active classes
          primaryProfile.classList.remove('active');
          comparisonProfile.classList.add('active');
          
          // Force reload the comparison profile if needed
          const testCase = document.getElementById('profile-test-case').value;
          const compareWith = compareWithSelect.value;
          
          if (testCase && compareWith) {
            // Use the global loadProfile function from profiles-data.js
            if (typeof window.loadProfile === 'function') {
              window.loadProfile(testCase, compareWith, comparisonProfile, true);
            } else if (typeof loadProfile === 'function') {
              loadProfile(testCase, compareWith, comparisonProfile, true);
            }
          }
        } else {
          // If no comparison is selected yet, show alert and keep primary active
          alert('Please select a profile to compare with first');
          
          // Reset button state
          tabButtons.forEach(btn => {
            if (btn.getAttribute('data-target') === 'primary-profile') {
              btn.classList.add('active');
            } else {
              btn.classList.remove('active');
            }
          });
        }
      }
    });
  });
  
  // Update tab visibility when comparison select changes
  compareWithSelect.addEventListener('change', function() {
    const comparisonTabBtn = document.querySelector('.profile-tab-btn[data-target="comparison-profile"]');
    
    if (this.value === '') {
      // If no comparison is selected, disable the comparison tab
      comparisonTabBtn.disabled = true;
      comparisonTabBtn.style.opacity = '0.5';
      
      // If comparison tab is active, switch to primary
      if (comparisonProfile.classList.contains('active')) {
        document.querySelector('.profile-tab-btn[data-target="primary-profile"]').click();
      }
    } else {
      // Enable comparison tab if a comparison is selected
      comparisonTabBtn.disabled = false;
      comparisonTabBtn.style.opacity = '1';
    }
  });
  
  // Initial setup of comparison tab state
  const comparisonTabBtn = document.querySelector('.profile-tab-btn[data-target="comparison-profile"]');
  if (compareWithSelect.value === '') {
    comparisonTabBtn.disabled = true;
    comparisonTabBtn.style.opacity = '0.5';
  }
}