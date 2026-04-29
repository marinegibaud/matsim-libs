package org.matsim.run;

import org.matsim.core.config.Config;
import org.matsim.core.config.ConfigUtils;
import org.matsim.core.controler.Controler;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * Calibration SPSA entierement automatisee pour MATSim - Scenario Sioux Falls
 *
 * Le programme :
 *   1. Injecte les parametres theta dans les 3 fichiers config XML
 *   2. Lance les 3 simulations MATSim en parallele via l'API Java
 *   3. Lit automatiquement les fichiers CSV de resultats
 *   4. Calcule le gradient et met a jour theta
 *   5. Repete jusqu'a convergence
 *
 * 5 parametres calibres :
 *   [0] constant_car   : config car > constant
 *   [1] money_car      : config car > monetaryDistanceRate   (<= 0)
 *   [2] U_pt           : config pt  > marginalUtilityOfTraveling_util_hr
 *   [3] constant_pt    : config pt  > constant
 *   [4] U_walk         : config walk > marginalUtilityOfTraveling_util_hr (<= 0)
 *
 * 9 observations extraites automatiquement :
 *   Parts modales  : modestats.csv      ligne 82, colonnes B/C/D
 *   Distances      : pkm_modestats.csv  ligne 82, colonnes B/C/D
 *   Temps          : ph_modestats.csv   ligne 82, colonnes B/C(+D)/H
 *
 * Hyperparametres SPSA : A=1, a=0.5, alpha=0.602, c=1, gamma=0.101
 *
 * Systeme de fichiers config :
 *   Iteration k (0-indexed) utilise les fichiers :
 *     config_1s{k+1}.xml  -> simulation theta courant
 *     config_2s{k+1}.xml  -> simulation theta+
 *     config_3s{k+1}.xml  -> simulation theta-
 *   Ces fichiers ne sont jamais reutilises entre iterations.
 */
public class SPSAAutomatique {

	// =========================================================================
	// CHEMINS — A ADAPTER SI NECESSAIRE
	// =========================================================================

	/** Dossier contenant les 30 fichiers config et les fichiers reseau/population */
	private static final String SCENARIO_DIR =
		"C:/Users/MII/Documents/matsim-projects/matsim-libsmultconfig/examples/scenarios/siouxfalls-2014";

	// =========================================================================
	// CONSTANTES
	// =========================================================================

	private static final String[] MODE_NAMES  = { "CAR", "PT", "WALK" };
	private static final String[] PARAM_NAMES = {
		"constant_car", "money_car", "U_pt", "constant_pt", "U_walk"
	};

	private static final int N_PARAMS = 5;
	private static final int N_MODES  = 3;
	private static final int N_OBS    = 9;   // 3 parts + 3 distances + 3 temps

	// Indices dans le vecteur d'observations
	private static final int IDX_PART_CAR   = 0;
	private static final int IDX_PART_PT    = 1;
	private static final int IDX_PART_WALK  = 2;
	private static final int IDX_DIST_CAR   = 3;
	private static final int IDX_DIST_PT    = 4;
	private static final int IDX_DIST_WALK  = 5;
	private static final int IDX_TEMPS_CAR  = 6;
	private static final int IDX_TEMPS_PT   = 7;
	private static final int IDX_TEMPS_WALK = 8;

	// Ligne cible dans les CSV MATSim
	private static final int CSV_TARGET_LINE = 80; // 0-indexed

	// =========================================================================
	// HYPERPARAMETRES SPSA
	// =========================================================================

	private static final double BIG_A  = 1.0;
	private static final double A_COEF = 0.5;
	private static final double ALPHA  = 0.602;
	private static final double C_COEF = 1.0;
	private static final double GAMMA  = 0.101;

	// =========================================================================
	// ETAT
	// =========================================================================

	private double[] theta;
	private final double[] observationsRef;
	private final int      maxIterations;
	private final String   saveFile;
	private final List<Double> objectiveHistory = new ArrayList<>();

	// =========================================================================
	// CONSTRUCTEUR
	// =========================================================================

	public SPSAAutomatique(double[] initialTheta,
						   double[] observationsRef,
						   int      maxIterations,
						   String   saveFile) {
		if (initialTheta.length   != N_PARAMS)
			throw new IllegalArgumentException("Attendu " + N_PARAMS + " parametres");
		if (observationsRef.length != N_OBS)
			throw new IllegalArgumentException("Attendu " + N_OBS + " observations");

		this.theta           = Arrays.copyOf(initialTheta,    N_PARAMS);
		this.observationsRef = Arrays.copyOf(observationsRef, N_OBS);
		this.maxIterations   = maxIterations;
		this.saveFile        = saveFile;
	}

	// =========================================================================
	// BOUCLE PRINCIPALE
	// =========================================================================

	public void calibrer() throws Exception {

		afficherEntete();
		int iterDebut = chargerEtat();

		for (int k = iterDebut; k < maxIterations; k++) {

			System.out.println("\n" + "=".repeat(65));
			System.out.println("  ITERATION " + k + " / " + (maxIterations - 1));
			System.out.println("=".repeat(65));

			// ------------------------------------------------------------------
			// ETAPE 1 : Pas ak et ck
			// ------------------------------------------------------------------
			double ak = A_COEF / Math.pow(k + BIG_A + 1.0, ALPHA);
			double ck = C_COEF / Math.pow(k + 1.0, GAMMA);
			System.out.printf("%nPas gradient  ak = %.6f%n", ak);
			System.out.printf("Perturbation  ck = %.6f%n", ck);

			// ------------------------------------------------------------------
			// ETAPE 2 : Vecteur delta (+-1, Bernoulli, seed = k)
			// ------------------------------------------------------------------
			Random rng = new Random(k);
			double[] delta = new double[N_PARAMS];
			for (int i = 0; i < N_PARAMS; i++)
				delta[i] = rng.nextBoolean() ? 1.0 : -1.0;

			System.out.print("Vecteur delta : [");
			for (int i = 0; i < N_PARAMS; i++)
				System.out.printf("%s%.0f", i > 0 ? ", " : "", delta[i]);
			System.out.println("]");

			// ------------------------------------------------------------------
			// ETAPE 3 : theta+  et  theta-
			// ------------------------------------------------------------------
			double[] thetaPlus  = perturber(theta, delta,  ck);
			double[] thetaMinus = perturber(theta, delta, -ck);
			projeterContraintes(thetaPlus);
			projeterContraintes(thetaMinus);

			afficherParametres(theta, thetaPlus, thetaMinus);

			// ------------------------------------------------------------------
// ETAPE 4 : Noms des fichiers config propres a cette iteration
// ------------------------------------------------------------------
			String configCenter = "config_1s" + (k + 1) + ".xml";
			String configPlus   = "config_2s" + (k + 1) + ".xml";
			String configMinus  = "config_3s" + (k + 1) + ".xml";

			String outCenter = outputDir("itera", k);
			String outPlus   = outputDir("iterb", k);
			String outMinus  = outputDir("iterc", k);

// Suppression des anciens dossiers output AVANT l'injection
			supprimerDossier(new File(outCenter));
			supprimerDossier(new File(outPlus));
			supprimerDossier(new File(outMinus));

// ------------------------------------------------------------------
// ETAPE 5 : Injection des parametres dans les 3 fichiers config
// ------------------------------------------------------------------
			System.out.println("\n[1/4] Injection des parametres dans les configs XML...");
			injecterConfig(configCenter, theta,      outCenter);
			injecterConfig(configPlus,   thetaPlus,  outPlus);
			injecterConfig(configMinus,  thetaMinus, outMinus);
			System.out.println("    OK — 3 configs mises a jour.");

			// ------------------------------------------------------------------
			// ETAPE 6 : Lancement des 3 simulations MATSim en parallele
			// ------------------------------------------------------------------
			System.out.println("[2/4] Lancement des 3 simulations MATSim en parallele...");
			long tDebut = System.currentTimeMillis();
			lancerSimulationsParallele(k);
			long duree = (System.currentTimeMillis() - tDebut) / 1000;
			System.out.printf("    OK — Simulations terminees en %d min %d s.%n",
				duree / 60, duree % 60);

			// ------------------------------------------------------------------
			// ETAPE 7 : Lecture des resultats CSV
			// ------------------------------------------------------------------
			System.out.println("[3/4] Lecture des resultats...");
			double[] resCenter = lireResultats(outCenter);
			double[] resPlus   = lireResultats(outPlus);
			double[] resMinus  = lireResultats(outMinus);
			afficherResultats("theta courant", resCenter);
			afficherResultats("theta+",        resPlus);
			afficherResultats("theta-",        resMinus);

			// ------------------------------------------------------------------
			// ETAPE 8 : Calcul des objectifs
			// ------------------------------------------------------------------
			double zCenter = calculerObjectif(resCenter);
			double zPlus   = calculerObjectif(resPlus);
			double zMinus  = calculerObjectif(resMinus);
			System.out.printf("%n  z(theta)  = %.5f%n", zCenter);
			System.out.printf("  z(theta+) = %.5f%n", zPlus);
			System.out.printf("  z(theta-) = %.5f%n", zMinus);
			objectiveHistory.add(zCenter);

			// ------------------------------------------------------------------
			// ETAPE 9 : Gradient et mise a jour
			// ------------------------------------------------------------------
			System.out.println("[4/4] Mise a jour de theta...");
			double[] grad = approximerGradient(zPlus, zMinus, ck, delta);
			double[] thetaAncien = Arrays.copyOf(theta, N_PARAMS);
			for (int i = 0; i < N_PARAMS; i++)
				theta[i] = thetaAncien[i] - ak * grad[i];
			projeterContraintes(theta);

			afficherMiseAJour(thetaAncien, theta, ak, grad);
			afficherIndicateurs(resCenter);
			sauvegarderEtat(k + 1);
		}

		afficherResumeFinal();
	}

	// =========================================================================
	// INJECTION DES PARAMETRES DANS LE CONFIG XML
	// =========================================================================

	/**
	 * Lit le fichier config_Ns{k+1}.xml depuis SCENARIO_DIR,
	 * injecte les parametres theta et l'outputDirectory,
	 * puis ecrit le fichier modifie dans le dossier output de la simulation.
	 *
	 * Le fichier source dans SCENARIO_DIR n'est JAMAIS supprime ni modifie :
	 * on lit, on modifie en memoire, on ecrit ailleurs.
	 *
	 * @param configFileName nom du fichier config source (ex: "config_1s3.xml")
	 * @param th             vecteur theta a injecter
	 * @param outputDir      chemin du dossier de sortie pour ce run
	 */
	private void injecterConfig(String configFileName, double[] th,
								String outputDir) throws Exception {

		// Lecture du fichier config source depuis SCENARIO_DIR
		String sourceConfigPath = SCENARIO_DIR + "/" + configFileName;

		// Parsing DOM
		javax.xml.parsers.DocumentBuilderFactory factory =
			javax.xml.parsers.DocumentBuilderFactory.newInstance();
		factory.setFeature(
			"http://apache.org/xml/features/nonvalidating/load-external-dtd", false);
		factory.setFeature(
			"http://xml.org/sax/features/validation", false);
		javax.xml.parsers.DocumentBuilder builder = factory.newDocumentBuilder();
		org.w3c.dom.Document doc = builder.parse(new File(sourceConfigPath));

		// --- 1. Mettre a jour outputDirectory ---
		setControllerParam(doc, "outputDirectory", outputDir);


		// --- 2. Mettre les chemins en absolus ---
		setModuleParam(doc, "network",    "inputNetworkFile",
			SCENARIO_DIR + "/Siouxfalls_network_PT.xml");
		setModuleParam(doc, "plans",      "inputPlansFile",
			SCENARIO_DIR + "/Siouxfalls_population.xml.gz");
		setModuleParam(doc, "facilities", "inputFacilitiesFile",
			SCENARIO_DIR + "/Siouxfalls_facilities.xml.gz");
		setModuleParam(doc, "transit",    "transitScheduleFile",
			SCENARIO_DIR + "/Siouxfalls_transitSchedule.xml");
		setModuleParam(doc, "transit",    "vehiclesFile",
			SCENARIO_DIR + "/Siouxfalls_vehicles.xml");

		// --- 3. Modifier uniquement les 5 parametres de scoring ---
		org.w3c.dom.NodeList paramsets = doc.getElementsByTagName("parameterset");
		for (int i = 0; i < paramsets.getLength(); i++) {
			org.w3c.dom.Element ps = (org.w3c.dom.Element) paramsets.item(i);
			if (!"modeParams".equals(ps.getAttribute("type"))) continue;

			String mode = getParamValue(ps, "mode");
			if (mode == null) continue;

			switch (mode) {
				case "car":
					setParamValue(ps, "constant",             th[0]);
					setParamValue(ps, "monetaryDistanceRate", th[1]);
					break;
				case "pt":
					setParamValue(ps, "marginalUtilityOfTraveling_util_hr", th[2]);
					setParamValue(ps, "constant",                           th[3]);
					break;
				case "walk":
					setParamValue(ps, "marginalUtilityOfTraveling_util_hr", th[4]);
					break;
			}
		}

		// --- 4. Ecriture dans le dossier output ---
		new File(outputDir).mkdirs();
		String destConfigPath = outputDir + "/" + configFileName;

		javax.xml.transform.TransformerFactory tf =
			javax.xml.transform.TransformerFactory.newInstance();
		javax.xml.transform.Transformer transformer = tf.newTransformer();
		transformer.setOutputProperty(
			javax.xml.transform.OutputKeys.INDENT, "yes");
		transformer.setOutputProperty(
			javax.xml.transform.OutputKeys.DOCTYPE_SYSTEM,
			"http://www.matsim.org/files/dtd/config_v2.dtd");
		transformer.transform(
			new javax.xml.transform.dom.DOMSource(doc),
			new javax.xml.transform.stream.StreamResult(new File(destConfigPath)));
	}

	// =========================================================================
	// HELPERS DOM
	// =========================================================================

	/** Trouve <module name="moduleName"> et change la valeur du <param name="paramName"> */
	private void setModuleParam(org.w3c.dom.Document doc,
								String moduleName, String paramName,
								String value) {
		org.w3c.dom.NodeList modules = doc.getElementsByTagName("module");
		for (int i = 0; i < modules.getLength(); i++) {
			org.w3c.dom.Element mod = (org.w3c.dom.Element) modules.item(i);
			if (!moduleName.equals(mod.getAttribute("name"))) continue;
			org.w3c.dom.NodeList params = mod.getElementsByTagName("param");
			for (int j = 0; j < params.getLength(); j++) {
				org.w3c.dom.Element p = (org.w3c.dom.Element) params.item(j);
				if (paramName.equals(p.getAttribute("name"))) {
					p.setAttribute("value", value);
					return;
				}
			}
		}
	}

	/** Cas special : outputDirectory et overwriteFiles sont dans <module name="controller"> */
	private void setControllerParam(org.w3c.dom.Document doc,
									String paramName, String value) {
		setModuleParam(doc, "controller", paramName, value);
	}

	/** Retourne la valeur d'un <param name="..."> dans un <parameterset> */
	private String getParamValue(org.w3c.dom.Element parameterset, String name) {
		org.w3c.dom.NodeList params = parameterset.getElementsByTagName("param");
		for (int i = 0; i < params.getLength(); i++) {
			org.w3c.dom.Element p = (org.w3c.dom.Element) params.item(i);
			if (name.equals(p.getAttribute("name")))
				return p.getAttribute("value");
		}
		return null;
	}

	/** Modifie la valeur d'un <param name="..."> dans un <parameterset> */
	private void setParamValue(org.w3c.dom.Element parameterset,
							   String name, double value) {
		org.w3c.dom.NodeList params = parameterset.getElementsByTagName("param");
		for (int i = 0; i < params.getLength(); i++) {
			org.w3c.dom.Element p = (org.w3c.dom.Element) params.item(i);
			if (name.equals(p.getAttribute("name"))) {
				p.setAttribute("value", String.valueOf(value));
				return;
			}
		}
	}

	// =========================================================================
	// LANCEMENT DES SIMULATIONS EN PARALLELE
	// =========================================================================

	/**
	 * Lance les 3 simulations MATSim dans des threads separes et attend
	 * que les 3 soient terminees avant de continuer.
	 *
	 * Chaque simulation utilise le fichier config_Ns{k+1}.xml copie dans
	 * son dossier output lors de l'injection.
	 */
	private void lancerSimulationsParallele(int k) throws Exception {

		String[] outDirs = {
			outputDir("itera", k),
			outputDir("iterb", k),
			outputDir("iterc", k)
		};

		String[] tempConfigs = {
			outDirs[0] + "/config_1s" + (k + 1) + ".xml",
			outDirs[1] + "/config_2s" + (k + 1) + ".xml",
			outDirs[2] + "/config_3s" + (k + 1) + ".xml"
		};

		// Plus de suppression ici, ni de verification d'existence :
		// les dossiers ont ete nettoyes et les configs injectees avant cet appel.

		ExecutorService pool = Executors.newFixedThreadPool(3);
		List<Future<?>> futures = new ArrayList<>();

		for (int i = 0; i < 3; i++) {
			final String configPath = tempConfigs[i];
			final int    simIndex   = i;

			futures.add(pool.submit(() -> {
				System.out.printf("    Simulation %d demarree (config: %s)%n",
					simIndex + 1, new File(configPath).getName());
				try {
					Config config = ConfigUtils.loadConfig(configPath);

					// Forcer l'ecrasement du dossier output directement via l'API Java,
					// independamment de ce qui est ecrit dans le XML.
					config.controller().setOverwriteFileSetting(
						org.matsim.core.controler.OutputDirectoryHierarchy
							.OverwriteFileSetting.overwriteExistingFiles);

					Controler controler = new Controler(config);
					controler.run();
					System.out.printf("    Simulation %d terminee.%n", simIndex + 1);
				} catch (Exception e) {
					throw new RuntimeException(
						"Simulation " + (simIndex + 1) + " echouee : " + e.getMessage(), e);
				}
			}));
		}

		pool.shutdown();
		boolean fini = pool.awaitTermination(4, TimeUnit.HOURS);
		if (!fini)
			throw new RuntimeException("Timeout : simulations trop longues (> 4h)");
		for (Future<?> f : futures) f.get();
	}
	/**
	 * Supprime recursivement un dossier et tout son contenu.
	 * Sans erreur si le dossier n'existe pas.
	 */
	private void supprimerDossier(File dossier) {
		if (!dossier.exists()) return;
		File[] fichiers = dossier.listFiles();
		if (fichiers != null) {
			for (File f : fichiers) {
				if (f.isDirectory()) supprimerDossier(f);
				else f.delete();
			}
		}
		dossier.delete();
	}

	// =========================================================================
	// LECTURE DES RESULTATS CSV
	// =========================================================================

	/**
	 * Lit les 3 fichiers CSV de sortie MATSim et retourne le vecteur
	 * de 9 observations.
	 */
	private double[] lireResultats(String outputDirPath) throws IOException {

		double[] obs = new double[N_OBS];

		// --- Parts modales ---
		double[] parts = lireColonnesCsv(
			outputDirPath + "/modestats.csv",
			CSV_TARGET_LINE,
			new int[]{ 1, 2, 3 });

		obs[IDX_PART_CAR]  = parts[0];
		obs[IDX_PART_PT]   = parts[1];
		obs[IDX_PART_WALK] = parts[2];

		// --- Distances (pkm -> metres) ---
		double[] pkm = lireColonnesCsv(
			outputDirPath + "/pkm_modestats.csv",
			CSV_TARGET_LINE,
			new int[]{ 1, 2, 3 });

		obs[IDX_DIST_CAR]  = pkm[0] ;
		obs[IDX_DIST_PT]   = pkm[1] ;
		obs[IDX_DIST_WALK] = pkm[2] ;

		// --- Temps (personne-heures -> secondes) ---
		double[] ph = lireColonnesCsv(
			outputDirPath + "/ph_modestats.csv",
			CSV_TARGET_LINE,
			new int[]{ 1, 3, 4, 7 });

		obs[IDX_TEMPS_CAR]  = ph[0] ;
		obs[IDX_TEMPS_PT]   = (ph[1] + ph[2]) ;
		obs[IDX_TEMPS_WALK] = ph[3] ;

		return obs;
	}

	/**
	 * Lit des colonnes specifiques d'une ligne cible dans un fichier CSV MATSim.
	 */
	private double[] lireColonnesCsv(String csvPath,
									 int    lineIndex,
									 int[]  colIndexes) throws IOException {

		double[] result = new double[colIndexes.length];

		try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
			String line;
			int    currentLine = -1;

			while ((line = br.readLine()) != null) {
				if (currentLine == lineIndex) {
					String[] tokens = line.split("[;\t]", -1);
					for (int i = 0; i < colIndexes.length; i++) {
						int col = colIndexes[i];
						if (col < tokens.length) {
							String val = tokens[col].trim();
							result[i] = val.isEmpty() ? 0.0 : Double.parseDouble(val);
						}
					}
					return result;
				}
				currentLine++;
			}
		}

		throw new IOException(
			"Ligne " + lineIndex + " introuvable dans : " + csvPath);
	}

	/**
	 * Construit le chemin absolu du dossier output pour une simulation.
	 */
	private String outputDir(String prefix, int k) {
		return SCENARIO_DIR + "/output/siouxfalls-2014-80iter" + prefix + (k + 1);
	}

	// =========================================================================
	// CALCULS SPSA
	// =========================================================================

	private double[] perturber(double[] base, double[] delta, double ckSigne) {
		double[] res = new double[N_PARAMS];
		for (int i = 0; i < N_PARAMS; i++)
			res[i] = base[i] + ckSigne * delta[i];
		return res;
	}

	private void projeterContraintes(double[] th) {
		if (th[0] > 0) th[0] = 0.0;
		if (th[1] > 0) th[1] = 0.0;
		if (th[2] > 0) th[2] = 0.0;
		if (th[3] > 0) th[3] = 0.0;
		if (th[4] > 0) th[4] = 0.0;
	}

	private double[] approximerGradient(double zPlus, double zMinus,
										double ck, double[] delta) {
		double diff = zPlus - zMinus;
		double[] grad = new double[N_PARAMS];
		for (int i = 0; i < N_PARAMS; i++)
			grad[i] = diff / (2.0 * ck * delta[i]);
		return grad;
	}

	private double calculerObjectif(double[] obs) {
		final double w1 = 1.0;
		final double w2 = 0.5;
		final double w3 = 0.5;

		double errPart = 0, errDist = 0, errTemps = 0;
		for (int m = 0; m < N_MODES; m++) {
			errPart += Math.abs(
				observationsRef[IDX_PART_CAR + m] - obs[IDX_PART_CAR + m]);

			double refD = observationsRef[IDX_DIST_CAR + m];
			if (refD > 0)
				errDist += Math.abs(refD - obs[IDX_DIST_CAR + m]) / refD;

			double refT = observationsRef[IDX_TEMPS_CAR + m];
			if (refT > 0)
				errTemps += Math.abs(refT - obs[IDX_TEMPS_CAR + m]) / refT;
		}
		return w1 * errPart + w2 * errDist + w3 * errTemps;
	}

	// =========================================================================
	// SAUVEGARDE / REPRISE
	// =========================================================================

	private void sauvegarderEtat(int prochIter) throws IOException {
		try (PrintWriter pw = new PrintWriter(new FileWriter(saveFile))) {
			pw.println("next_iteration," + String.join(",", PARAM_NAMES));
			StringBuilder sb = new StringBuilder().append(prochIter);
			for (double v : theta) sb.append(",").append(v);
			pw.println(sb);
		}
		System.out.printf("  [Sauvegarde -> %s  (prochaine iter : %d)]%n",
			saveFile, prochIter);
	}

	private int chargerEtat() throws IOException {
		File f = new File(saveFile);
		if (!f.exists()) return 0;

		Scanner sc = new Scanner(System.in);
		System.out.print("\nFichier de sauvegarde detecte. Reprendre ? (o/n) : ");
		if (!sc.nextLine().trim().equalsIgnoreCase("o")) return 0;

		try (BufferedReader br = new BufferedReader(new FileReader(f))) {
			br.readLine();
			String[] parts = br.readLine().split(",");
			int iterDebut = Integer.parseInt(parts[0]);
			for (int i = 0; i < N_PARAMS; i++)
				theta[i] = Double.parseDouble(parts[i + 1]);
			System.out.println("Reprise a l'iteration " + iterDebut);
			afficherTheta(theta);
			return iterDebut;
		}
	}

	// =========================================================================
	// AFFICHAGE
	// =========================================================================

	private void afficherEntete() {
		System.out.println("\n" + "=".repeat(65));
		System.out.println("   CALIBRATION SPSA AUTOMATIQUE — MATSIM SIOUX FALLS");
		System.out.println("=".repeat(65));
		System.out.println("  Scenario  : " + SCENARIO_DIR);
		System.out.printf("  SPSA      : A=%.0f  a=%.3f  alpha=%.3f  c=%.0f  gamma=%.3f%n",
			BIG_A, A_COEF, ALPHA, C_COEF, GAMMA);
		System.out.println("  Configs   : config_Ns{iter+1}.xml  (N=1 theta, 2 theta+, 3 theta-)");
		System.out.println("  Contraintes : money_car <= 0  |  U_walk <= 0");
		System.out.println("\n  Theta initial :");
		afficherTheta(theta);
		System.out.println("\n  Observations de reference :");
		afficherObsRef();
	}

	private void afficherParametres(double[] th, double[] thP, double[] thM) {
		System.out.println("\n" + "-".repeat(65));
		System.out.println("  PARAMETRES INJECTES DANS LES 3 CONFIGS");
		System.out.println("-".repeat(65));
		System.out.printf("  %-16s  %12s  %12s  %12s%n",
			"Parametre", "theta", "theta+", "theta-");
		System.out.println("  " + "-".repeat(57));
		for (int i = 0; i < N_PARAMS; i++)
			System.out.printf("  %-16s  %+12.5f  %+12.5f  %+12.5f%n",
				PARAM_NAMES[i], th[i], thP[i], thM[i]);
	}

	private void afficherResultats(String label, double[] obs) {
		System.out.printf("%n  Resultats [%s] :%n", label);
		System.out.printf("  %-5s  %-12s  %-14s  %-12s%n",
			"Mode", "Part modale", "Distance (m)", "Temps (s)");
		System.out.println("  " + "-".repeat(48));
		for (int m = 0; m < N_MODES; m++)
			System.out.printf("  %-5s  %-12.4f  %-14.1f  %-12.1f%n",
				MODE_NAMES[m],
				obs[IDX_PART_CAR + m],
				obs[IDX_DIST_CAR + m],
				obs[IDX_TEMPS_CAR + m]);
	}

	private void afficherMiseAJour(double[] ancien, double[] nouveau,
								   double ak, double[] grad) {
		System.out.println("\n  Mise a jour de theta :");
		System.out.printf("  %-16s  %10s  %12s  %10s  %s%n",
			"Parametre", "ancien", "correction", "nouveau", "contrainte");
		System.out.println("  " + "-".repeat(62));
		String[] contraintes = { "libre", "<= 0", "libre", "libre", "<= 0" };
		for (int i = 0; i < N_PARAMS; i++) {
			System.out.printf("  %-16s  %+10.5f  %+12.5f  %+10.5f  %s%n",
				PARAM_NAMES[i], ancien[i], -ak * grad[i], nouveau[i],
				contraintes[i]);
		}
	}

	private void afficherIndicateurs(double[] obs) {
		System.out.println("\n  Indicateurs de performance :");
		double rmsn = 0, mpe = 0, mane = 0;
		for (int m = 0; m < N_MODES; m++) {
			double ref = observationsRef[IDX_PART_CAR + m];
			double sim = obs[IDX_PART_CAR + m];
			double err = sim - ref;
			rmsn += err * err;
			if (ref > 0) { mpe += err / ref; mane += Math.abs(err / ref); }
		}
		rmsn = Math.sqrt(rmsn / N_MODES);
		System.out.printf("    RMSN = %.4f  |  MPE = %.4f  |  MANE = %.4f%n",
			rmsn, mpe / N_MODES, mane / N_MODES);
	}

	private void afficherTheta(double[] th) {
		for (int i = 0; i < N_PARAMS; i++)
			System.out.printf("    %-16s = %+.5f%n", PARAM_NAMES[i], th[i]);
	}

	private void afficherObsRef() {
		System.out.printf("  %-5s  %-12s  %-14s  %-12s%n",
			"Mode", "Part ref", "Dist ref (m)", "Temps ref (s)");
		System.out.println("  " + "-".repeat(48));
		for (int m = 0; m < N_MODES; m++)
			System.out.printf("  %-5s  %-12.4f  %-14.1f  %-12.1f%n",
				MODE_NAMES[m],
				observationsRef[IDX_PART_CAR + m],
				observationsRef[IDX_DIST_CAR + m],
				observationsRef[IDX_TEMPS_CAR + m]);
	}

	private void afficherResumeFinal() {
		System.out.println("\n" + "=".repeat(65));
		System.out.println("  CALIBRATION TERMINEE");
		System.out.println("=".repeat(65));
		System.out.println("\n  Theta calibre final :");
		afficherTheta(theta);
		System.out.println("\n  Historique de z(theta) :");
		for (int i = 0; i < objectiveHistory.size(); i++)
			System.out.printf("    Iter %3d : %.5f%n", i, objectiveHistory.get(i));
	}

	// =========================================================================
	// MAIN
	// =========================================================================

	public static void main(String[] args) throws Exception {

		double[] initialTheta = {
			-0.02534,   // constant_car
			0.0,       // money_car      (<= 0)
			-0.27180,   // U_pt
			-0.51416,   // constant_pt
			-0.66755    // U_walk         (<= 0)
		};

		double[] observationsRef = {
			0.467614,   // part CAR
			0.269778,  // part PT
			0.262608,  // part WALK
			418314.0,            // distance CAR  (m)
			123280.0,            // distance PT   (m)
			251798.0,            // distance WALK (m)
			9200.0,             // temps CAR  (s)
			7698.0,             // temps PT   (s)
			83896.0              // temps WALK (s)
		};

		SPSAAutomatique calibrateur = new SPSAAutomatique(
			initialTheta,
			observationsRef,
			25,                      // nombre max d'iterations SPSA
			"spsa_sauvegarde.csv"    // fichier de reprise
		);

		calibrateur.calibrer();
	}
}
