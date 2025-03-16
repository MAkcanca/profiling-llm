// Gold Standards data - this is populated dynamically by the GoldStandardsLoader
let goldStandardsData = [];

// Initialize the page when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', async function() {
    // Initialize theme
    initTheme();
    
    // Load the gold standards data dynamically
    await loadGoldStandardsData();
    
    // Generate gold standard cards only once
    const container = document.getElementById('gold-standards-grid');
    if (container && container.children.length === 0) {
        generateGoldStandardCards();
    }
    
    // Set up event listeners
    setupEventListeners();
    
    // Check if a specific profile should be loaded from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const profileId = urlParams.get('profile');
    if (profileId) {
        const profile = goldStandardsData.find(p => p.id === profileId);
        if (profile) {
            loadProfile(profile);
        }
    }
});

// Load gold standards data using the dynamic loader
async function loadGoldStandardsData() {
    try {
        // Check if the GoldStandardsLoader is available
        if (window.GoldStandardsLoader) {
            goldStandardsData = await window.GoldStandardsLoader.loadGoldStandards();
        } else {
            console.error('GoldStandardsLoader not found. Make sure gold-standards-loader.js is loaded.');
            goldStandardsData = [];
        }
        
        if (goldStandardsData.length === 0) {
            // Display error message in the container
            const container = document.getElementById('gold-standards-grid');
            if (container) {
                container.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-triangle"></i>
                        <p>No gold standard profiles were found.</p>
                        <p>Please make sure the gold-standards directory contains valid profile JSON files.</p>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Error loading gold standards data:', error);
        goldStandardsData = [];
    }
}

// Theme toggle functionality (reused from the main dashboard)
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

// Generate cards for each gold standard profile
function generateGoldStandardCards() {
    const container = document.getElementById('gold-standards-grid');
    if (!container) return;
    
    // Check if cards are already generated (to prevent duplicates)
    if (container.querySelector('.gold-standard-card')) {
        console.log('Gold standard cards already generated, skipping');
        return;
    }
    
    // Clear the container
    container.innerHTML = '';
    
    // If no data is available, show a message
    if (goldStandardsData.length === 0) {
        container.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>No gold standard profiles were found.</p>
                <p>Please make sure the gold-standards directory contains valid profile JSON files.</p>
            </div>
        `;
        return;
    }
    
    // Create a Set to track unique profile IDs
    const addedProfileIds = new Set();
    
    goldStandardsData.forEach(profile => {
        // Skip if we've already added this profile ID
        if (addedProfileIds.has(profile.id)) {
            console.log(`Skipping duplicate profile: ${profile.id}`);
            return;
        }
        
        // Add to our tracking set
        addedProfileIds.add(profile.id);
        
        const card = document.createElement('div');
        card.className = 'gold-standard-card';
        card.setAttribute('data-id', profile.id);
        
        // Create a fallback image URL if the image doesn't exist
        const imageUrl = profile.image || `https://via.placeholder.com/400x250/3498db/ffffff?text=${encodeURIComponent(profile.name)}`;
        
        card.innerHTML = `
            <div class="card-image" style="background-image: url('${imageUrl}')"></div>
            <div class="card-content">
                <h3 class="card-title">${profile.name}</h3>
                <div class="card-subtitle">${profile.subtitle}</div>
                <div class="card-tags">
                    <span class="card-tag framework-nas">${formatFrameworkName('nas', profile.frameworks.nas)}</span>
                    <span class="card-tag framework-shpa">${formatFrameworkName('shpa', profile.frameworks.shpa)}</span>
                    <span class="card-tag framework-bcs">${formatFrameworkName('bcs', profile.frameworks.bcs)}</span>
                    <span class="card-tag framework-spatial">${formatFrameworkName('spatial', profile.frameworks.spatial)}</span>
                    ${profile.tags && profile.tags.length > 0 ? profile.tags.map(tag => `<span class="card-tag">${tag}</span>`).join('') : ''}
                </div>
                <div class="card-description">${profile.description}</div>
            </div>
        `;
        
        // Add click event to open the profile viewer
        card.addEventListener('click', () => {
            loadProfile(profile);
            // Update URL without refreshing the page
            window.history.pushState({}, '', `?profile=${profile.id}`);
        });
        
        container.appendChild(card);
    });
}

// Format framework names for display
function formatFrameworkName(framework, value) {
    if (value === 'NOT_APPLICABLE') return 'N/A';
    
    switch (framework) {
        case 'nas':
            return value.split('_').map(word => word.charAt(0) + word.slice(1).toLowerCase()).join(' ');
        case 'shpa':
            return value.split('_').map(word => word.charAt(0) + word.slice(1).toLowerCase()).join(' ');
        case 'bcs':
            return value.charAt(0) + value.slice(1).toLowerCase();
        case 'spatial':
            return value.charAt(0) + value.slice(1).toLowerCase();
        default:
            return value;
    }
}

// Set up event listeners
function setupEventListeners() {
    // Close button for profile viewer
    const closeButton = document.getElementById('close-profile');
    if (closeButton) {
        closeButton.addEventListener('click', () => {
            document.getElementById('profile-viewer').classList.remove('active');
            // Remove profile from URL
            window.history.pushState({}, '', window.location.pathname);
        });
    }
    
    // Tab navigation in profile viewer
    document.querySelectorAll('.tab-btn').forEach(tab => {
        tab.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            
            // Remove active class from all tabs and tab panes
            document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            
            // Add active class to selected tab and tab pane
            this.classList.add('active');
            document.querySelector(`.tab-pane[data-tab="${tabName}"]`).classList.add('active');
        });
    });
    
    // Close profile viewer when clicking outside content
    const profileViewer = document.getElementById('profile-viewer');
    if (profileViewer) {
        profileViewer.addEventListener('click', function(e) {
            if (e.target === this) {
                this.classList.remove('active');
                // Remove profile from URL
                window.history.pushState({}, '', window.location.pathname);
            }
        });
    }
    
    // Handle browser back/forward navigation
    window.addEventListener('popstate', function() {
        const urlParams = new URLSearchParams(window.location.search);
        const profileId = urlParams.get('profile');
        
        if (profileId) {
            const profile = goldStandardsData.find(p => p.id === profileId);
            if (profile) {
                loadProfile(profile);
            }
        } else {
            const profileViewer = document.getElementById('profile-viewer');
            if (profileViewer) {
                profileViewer.classList.remove('active');
            }
        }
    });
}

// Load and display a profile
function loadProfile(profile) {
    const profileViewer = document.getElementById('profile-viewer');
    const profileTitle = document.getElementById('profile-title');
    const tabContent = document.getElementById('tab-content');
    
    if (!profileViewer || !profileTitle || !tabContent) {
        console.error('Profile viewer elements not found in the DOM');
        return;
    }
    
    // Set profile title
    profileTitle.textContent = `${profile.name} - ${profile.subtitle}`;
    
    // Show loading indicator
    tabContent.innerHTML = '<div class="loading-indicator"><i class="fas fa-spinner fa-spin"></i> Loading profile...</div>';
    
    // Set profile image
    const profileImage = document.getElementById('profile-image');
    const imageUrl = profile.image || `https://via.placeholder.com/350x300/3498db/ffffff?text=${encodeURIComponent(profile.name)}`;
    
    if (profileImage) {
        profileImage.innerHTML = `<img src="${imageUrl}" alt="${profile.name}" onerror="this.onerror=null; this.src='https://via.placeholder.com/350x300/3498db/ffffff?text=${encodeURIComponent(profile.name)}';">`;
    }
    
    // Activate profile viewer
    profileViewer.classList.add('active');
    
    // Fetch the profile data
    fetch(profile.filePath)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error loading profile: ${response.status} ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            // Populate the case summary
            populateCaseSummary(profile, data);
            
            // Populate framework classifications
            populateFrameworkClassifications(data);
            
            // Generate tab content
            generateTabContent(data);
            
            // Activate the first tab by default
            const firstTab = document.querySelector('.tab-btn');
            if (firstTab) {
                firstTab.click();
            }
        })
        .catch(error => {
            console.error('Error loading profile:', error);
            tabContent.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Error loading profile: ${error.message}</p>
                </div>
            `;
        });
}

// Populate the case summary section
function populateCaseSummary(profile, data) {
    const caseSummary = document.getElementById('case-summary');
    if (!caseSummary) return;
    
    caseSummary.innerHTML = `
        <p><strong>Case ID:</strong> ${data.case_id || 'N/A'}</p>
        <p>${profile.description}</p>
        <div class="case-tags">
            ${profile.tags && profile.tags.length > 0 ? 
              profile.tags.map(tag => `<span class="card-tag">${tag}</span>`).join('') : ''}
            <span class="card-tag framework-nas">${formatFrameworkName('nas', profile.frameworks.nas)}</span>
            <span class="card-tag framework-shpa">${formatFrameworkName('shpa', profile.frameworks.shpa)}</span>
            <span class="card-tag framework-bcs">${formatFrameworkName('bcs', profile.frameworks.bcs)}</span>
            <span class="card-tag framework-spatial">${formatFrameworkName('spatial', profile.frameworks.spatial)}</span>
        </div>
    `;
}

// Populate the framework classifications section
function populateFrameworkClassifications(data) {
    const frameworkClassifications = document.getElementById('framework-classifications');
    if (!frameworkClassifications) return;
    
    const frameworks = data.framework_classifications || {};
    
    let html = '';
    
    // Narrative Action System
    if (frameworks.narrative_action_system) {
        html += createFrameworkItem(
            'nas',
            'Narrative Action System',
            frameworks.narrative_action_system.primary_classification,
            frameworks.narrative_action_system.confidence,
            frameworks.narrative_action_system.supporting_evidence,
            frameworks.narrative_action_system.contradicting_evidence,
            frameworks.narrative_action_system.reasoning
        );
    }
    
    // Sexual Behavioral Analysis
    if (frameworks.sexual_behavioral_analysis) {
        html += createFrameworkItem(
            'shpa',
            'Sexual Behavioral Analysis',
            frameworks.sexual_behavioral_analysis.primary_classification,
            frameworks.sexual_behavioral_analysis.confidence,
            frameworks.sexual_behavioral_analysis.supporting_evidence,
            frameworks.sexual_behavioral_analysis.contradicting_evidence,
            frameworks.sexual_behavioral_analysis.reasoning
        );
    }
    
    // Sexual Homicide Pathways Analysis
    if (frameworks.sexual_homicide_pathways_analysis) {
        html += createFrameworkItem(
            'bcs',
            'Sexual Homicide Pathways',
            frameworks.sexual_homicide_pathways_analysis.primary_classification,
            frameworks.sexual_homicide_pathways_analysis.confidence,
            frameworks.sexual_homicide_pathways_analysis.supporting_evidence,
            frameworks.sexual_homicide_pathways_analysis.contradicting_evidence,
            frameworks.sexual_homicide_pathways_analysis.reasoning
        );
    } 
    // Behavioral Change Staging (older format)
    else if (frameworks.behavioral_change_staging) {
        html += createFrameworkItem(
            'bcs',
            'Behavioral Change Staging',
            frameworks.behavioral_change_staging.primary_classification,
            frameworks.behavioral_change_staging.confidence,
            frameworks.behavioral_change_staging.supporting_evidence,
            frameworks.behavioral_change_staging.contradicting_evidence,
            frameworks.behavioral_change_staging.reasoning
        );
    }
    
    // Spatial Behavioral Analysis
    if (frameworks.spatial_behavioral_analysis) {
        html += createFrameworkItem(
            'spatial',
            'Spatial Behavioral Analysis',
            frameworks.spatial_behavioral_analysis.primary_classification,
            frameworks.spatial_behavioral_analysis.confidence,
            frameworks.spatial_behavioral_analysis.supporting_evidence,
            frameworks.spatial_behavioral_analysis.contradicting_evidence,
            frameworks.spatial_behavioral_analysis.reasoning
        );
    }
    
    frameworkClassifications.innerHTML = html || '<p>No framework classifications available</p>';
}

// Create a framework item HTML
function createFrameworkItem(type, name, classification, confidence, supporting, contradicting, reasoning) {
    // Format classification and confidence display, handling null/undefined values
    let classificationDisplay = classification ? classification : 'Not Available';
    let confidenceDisplay = confidence !== null && confidence !== undefined ? 
        `(${(confidence * 100).toFixed(0)}%)` : '';
    
    return `
        <div class="framework-item">
            <div class="framework-name framework-${type}">
                ${name}
                <span class="framework-value">${classificationDisplay} ${confidenceDisplay}</span>
            </div>
            ${supporting && supporting.length > 0 ? `
                <div class="framework-evidence">
                    <div class="evidence-title">Supporting Evidence:</div>
                    <ul class="evidence-list">
                        ${supporting.map(evidence => `<li>${evidence}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            ${contradicting && contradicting.length > 0 ? `
                <div class="framework-evidence">
                    <div class="evidence-title">Contradicting Evidence:</div>
                    <ul class="evidence-list">
                        ${contradicting.map(evidence => `<li>${evidence}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            ${reasoning && reasoning.length > 0 ? `
                <div class="framework-evidence">
                    <div class="evidence-title">Reasoning:</div>
                    <ul class="evidence-list">
                        ${reasoning.map(reason => `<li>${reason}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        </div>
    `;
}

// Generate tab content based on profile data
function generateTabContent(data) {
    const tabContent = document.getElementById('tab-content');
    if (!tabContent) return;
    
    const profile = data.offender_profile || {};
    
    // Create tab panes for each section
    let html = '';
    
    // Demographics tab
    html += createTabPane('demographics', 'Demographics', profile.demographics);
    
    // Psychological tab
    html += createTabPane('psychological', 'Psychological Characteristics', profile.psychological_characteristics);
    
    // Behavioral tab
    html += createTabPane('behavioral', 'Behavioral Characteristics', profile.behavioral_characteristics);
    
    // Geographic tab
    html += createTabPane('geographic', 'Geographic Behavior', profile.geographic_behavior);
    
    // Skills tab
    html += createTabPane('skills', 'Skills & Knowledge', profile.skills_and_knowledge);
    
    // Investigative tab
    html += createTabPane('investigative', 'Investigative Implications', profile.investigative_implications);
    
    // Identifiers tab
    html += createTabPane('identifiers', 'Key Identifiers', profile.key_identifiers);
    
    // Validation tab
    if (data.validation_metrics) {
        html += `
            <div class="tab-pane" data-tab="validation">
                <div class="content-section">
                    <h3>Validation Metrics</h3>
                    ${data.validation_metrics.key_behavioral_indicators ? `
                        <div class="content-item">
                            <div class="content-item-label">Key Behavioral Indicators</div>
                            <ul class="evidence-list">
                                ${data.validation_metrics.key_behavioral_indicators.map(item => `<li>${item}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    ${data.validation_metrics.critical_evidence ? `
                        <div class="content-item">
                            <div class="content-item-label">Critical Evidence</div>
                            <ul class="evidence-list">
                                ${data.validation_metrics.critical_evidence.map(item => `<li>${item}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    ${data.validation_metrics.profile_accuracy_factors ? `
                        <div class="content-item">
                            <div class="content-item-label">Profile Accuracy Factors</div>
                            <ul class="evidence-list">
                                ${data.validation_metrics.profile_accuracy_factors.map(item => `<li>${item}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    tabContent.innerHTML = html || '<p>No profile data available</p>';
}

// Create a tab pane for a section of the profile
function createTabPane(tabId, title, sectionData) {
    if (!sectionData) return '';
    
    let html = `<div class="tab-pane" data-tab="${tabId}"><div class="content-section"><h3>${title}</h3>`;
    
    // Process each field in the section
    for (const [key, value] of Object.entries(sectionData)) {
        if (key === 'reasoning') continue; // Handle reasoning separately
        
        html += `
            <div class="content-item">
                <div class="content-item-label">${formatFieldName(key)}</div>
                <div class="content-item-value">${value}</div>
            </div>
        `;
    }
    
    // Add reasoning if available
    if (sectionData.reasoning && sectionData.reasoning.length > 0) {
        html += `
            <div class="reasoning-section">
                <div class="reasoning-title">Reasoning:</div>
                <ul class="reasoning-list">
                    ${sectionData.reasoning.map(reason => `<li>${reason}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    html += '</div></div>';
    return html;
}

// Format field names for display
function formatFieldName(fieldName) {
    return fieldName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}